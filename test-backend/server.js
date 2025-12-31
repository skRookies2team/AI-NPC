const express = require('express');
const multer = require('multer');
const axios = require('axios');
const cors = require('cors');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const { S3Client, PutObjectCommand, GetObjectCommand } = require('@aws-sdk/client-s3');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');
const crypto = require('crypto');

const app = express();
const upload = multer({ dest: 'uploads/' }); // Node 서버 임시 저장 경로 (기존 방식용)

app.use(cors());
app.use(express.json());

const AI_SERVER_URL = process.env.AI_SERVER_URL || 'http://localhost:8002'; // Python 서버 주소

// S3 설정 (환경변수에서 가져오기)
const S3_BUCKET = process.env.S3_BUCKET;
const S3_REGION = process.env.S3_REGION || 'ap-northeast-2';
const AWS_ACCESS_KEY_ID = process.env.AWS_ACCESS_KEY_ID;
const AWS_SECRET_ACCESS_KEY = process.env.AWS_SECRET_ACCESS_KEY;

// S3 클라이언트 초기화
let s3Client = null;
if (S3_BUCKET && AWS_ACCESS_KEY_ID && AWS_SECRET_ACCESS_KEY) {
    s3Client = new S3Client({
        region: S3_REGION,
        credentials: {
            accessKeyId: AWS_ACCESS_KEY_ID,
            secretAccessKey: AWS_SECRET_ACCESS_KEY,
        },
    });
    console.log('S3 클라이언트 초기화 완료');
} else {
    console.warn('S3 설정이 없습니다. 환경변수를 확인해주세요.');
}

// 1. S3 Presigned URL 생성 (프론트엔드에서 S3에 직접 업로드하기 위한 URL)
app.post('/api/s3/presigned-url', async (req, res) => {
    try {
        if (!s3Client) {
            return res.status(500).json({ 
                success: false, 
                message: 'S3 설정이 되어있지 않습니다.' 
            });
        }

        const { filename, contentType = 'text/plain' } = req.body;
        if (!filename) {
            return res.status(400).json({ 
                success: false, 
                message: '파일명이 필요합니다.' 
            });
        }

        // 세션 ID 생성
        const session_id = 'sess_' + Date.now();
        
        // S3 키 생성 (세션 ID를 포함하여 고유하게)
        const fileKey = `novels/${session_id}/${filename}`;

        // Presigned URL 생성 (업로드용, 5분 유효)
        const putCommand = new PutObjectCommand({
            Bucket: S3_BUCKET,
            Key: fileKey,
            ContentType: contentType,
        });

        const presignedUrl = await getSignedUrl(s3Client, putCommand, { expiresIn: 300 });

        res.json({
            success: true,
            presigned_url: presignedUrl,
            file_key: fileKey,
            session_id: session_id,
            bucket: S3_BUCKET
        });

    } catch (error) {
        console.error('S3 Presigned URL 생성 오류:', error.message);
        res.status(500).json({ 
            success: false, 
            message: 'Presigned URL 생성 실패: ' + error.message 
        });
    }
});

// 2. S3 업로드 완료 후 학습 요청 (프론트엔드에서 S3 업로드 완료 후 호출)
app.post('/api/train-from-s3', async (req, res) => {
    try {
        const { session_id, file_key, character_name } = req.body;

        if (!session_id || !file_key || !character_name) {
            return res.status(400).json({ 
                success: false, 
                message: 'session_id, file_key, character_name이 모두 필요합니다.' 
            });
        }

        // AI 서버로 S3 파일 정보 전달하여 학습 요청
        const response = await axios.post(`${AI_SERVER_URL}/api/ai/train-from-s3`, {
            session_id: session_id,
            file_key: file_key,
            bucket: S3_BUCKET,
            character_name: character_name
        });

        res.json({
            success: true,
            message: '캐릭터 생성 완료!',
            session_id: session_id
        });

    } catch (error) {
        console.error('학습 요청 오류:', error.message);
        res.status(500).json({ 
            success: false, 
            message: '학습 중 오류 발생: ' + (error.response?.data?.detail || error.message) 
        });
    }
});

// 3. 기존 방식 (하위 호환성 유지 - 테스트용)
// 주의: 이 엔드포인트는 테스트용이며, 실제 운영에서는 S3 방식을 사용하세요
// character_name 없이 소설만 학습 (실제 서비스 환경처럼)
app.post('/upload', upload.single('file'), async (req, res) => {
    try {
        const file = req.file;

        if (!file) return res.status(400).json({ success: false, message: "파일이 없습니다." });

        // 세션 ID 생성
        const session_id = 'sess_' + Date.now();
        
        // 파일 내용 읽기
        const fileContent = fs.readFileSync(file.path, 'utf-8');
        
        // 원본 임시 파일 삭제
        fs.unlinkSync(file.path);

        // AI 서버로 텍스트 내용 직접 전송 (테스트용)
        // character_name 없이 소설만 학습 (나중에 /api/ai/character에서 설정)
        const response = await axios.post(`${AI_SERVER_URL}/api/ai/train-text`, {
            session_id: session_id,
            content: fileContent,
            character_name: ""  // 빈 값으로 전송 (나중에 character 엔드포인트에서 설정)
        });

        res.json({ 
            success: true, 
            message: '소설 학습 완료! 이제 캐릭터를 설정하세요.', 
            session_id: session_id 
        });

    } catch (error) {
        console.error('Upload Error:', error.message);
        if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
        res.status(500).json({ 
            success: false, 
            message: '학습 중 오류 발생: ' + error.message 
        });
    }
});

// 4. 캐릭터 설정 엔드포인트 (실제 서비스 환경)
app.post('/api/character', async (req, res) => {
    try {
        const { session_id, character_name, character_description } = req.body;

        if (!session_id || !character_name) {
            return res.status(400).json({ 
                success: false, 
                message: 'session_id와 character_name이 필요합니다.' 
            });
        }

        // AI 서버로 캐릭터 설정 요청
        const response = await axios.post(`${AI_SERVER_URL}/api/ai/character`, {
            session_id: session_id,
            character_name: character_name,
            character_description: character_description || ""
        });

        res.json({
            success: true,
            message: '캐릭터 설정 완료!',
            ...response.data
        });

    } catch (error) {
        console.error('Character 설정 오류:', error.message);
        res.status(500).json({ 
            success: false, 
            message: '캐릭터 설정 중 오류 발생: ' + (error.response?.data?.detail || error.message) 
        });
    }
});

// 2. 채팅 요청 (API Key 불필요)
app.post('/chat', async (req, res) => {
    try {
        const { message, session_id, character_name } = req.body;

        if (!message || !session_id || !character_name) {
            return res.status(400).json({ 
                success: false, 
                message: 'message, session_id, character_name이 모두 필요합니다.' 
            });
        }

        // AI 서버로 메시지 전달
        const response = await axios.post(`${AI_SERVER_URL}/api/ai/chat`, {
            message,
            session_id,
            character_name
        });

        res.json(response.data);

    } catch (error) {
        console.error('Chat Error:', error.message);
        if (error.response) {
            console.error('AI Server Response:', error.response.data);
            return res.status(error.response.status).json({ 
                reply: error.response.data.detail || '오류가 발생했습니다.' 
            });
        }
        res.status(500).json({ reply: '오류가 발생했습니다.' });
    }
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Main Backend running on http://localhost:${PORT}`);
});