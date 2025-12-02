const express = require('express');
const multer = require('multer');
const axios = require('axios');
const cors = require('cors');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' }); // Node 서버 임시 저장 경로

app.use(cors());
app.use(express.json());

const AI_SERVER_URL = 'http://localhost:8000'; // Python 서버 주소

// 1. 업로드 및 학습 요청 (API Key 불필요)
app.post('/upload', upload.single('file'), async (req, res) => {
    try {
        const { character_name } = req.body;
        const file = req.file;

        if (!file) return res.status(400).json({ success: false, message: "파일이 없습니다." });

        // 세션 ID 생성
        const session_id = 'sess_' + Date.now();

        // Python 서버로 전송할 폼 데이터 구성
        const formData = new FormData();
        formData.append('file', fs.createReadStream(file.path));
        formData.append('character_name', character_name);
        formData.append('session_id', session_id);

        // AI 서버로 요청 (API Key는 AI 서버가 가지고 있음)
        const response = await axios.post(`${AI_SERVER_URL}/api/ai/train`, formData, {
            headers: { ...formData.getHeaders() }
        });

        // 임시 파일 정리
        fs.unlinkSync(file.path);

        res.json({ 
            success: true, 
            message: '캐릭터 생성 완료!', 
            session_id: session_id 
        });

    } catch (error) {
        console.error('AI Server Error:', error.message);
        if(req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
        res.status(500).json({ success: false, message: '학습 중 오류 발생' });
    }
});

// 2. 채팅 요청 (API Key 불필요)
app.post('/chat', async (req, res) => {
    try {
        const { message, session_id } = req.body;

        // AI 서버로 메시지 전달
        const response = await axios.post(`${AI_SERVER_URL}/api/ai/chat`, {
            message,
            session_id
        });

        res.json(response.data);

    } catch (error) {
        console.error('Chat Error:', error.message);
        res.status(500).json({ reply: '오류가 발생했습니다.' });
    }
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Main Backend running on http://localhost:${PORT}`);
});