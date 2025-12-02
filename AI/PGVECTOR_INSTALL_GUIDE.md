# PostgreSQL 18.1 + pgvector 설치 가이드 (Windows)

## 방법 1: WSL2 사용 (가장 확실함)

### 1. WSL2 설치 (관리자 권한 PowerShell)
```powershell
wsl --install
```

### 2. WSL Ubuntu에서 PostgreSQL + pgvector 설치
```bash
# PostgreSQL 18 설치
sudo apt update
sudo apt install -y postgresql-18 postgresql-server-dev-18 git build-essential

# pgvector 설치
cd /tmp
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# PostgreSQL 시작
sudo service postgresql start

# 데이터베이스 생성
sudo -u postgres psql -c "CREATE DATABASE ai_npc_db;"
sudo -u postgres psql -d ai_npc_db -c "CREATE EXTENSION vector;"

# 비밀번호 설정
sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'yourpassword';"

# 외부 접속 허용 (선택)
sudo sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/g" /etc/postgresql/18/main/postgresql.conf
echo "host all all 0.0.0.0/0 md5" | sudo tee -a /etc/postgresql/18/main/pg_hba.conf
sudo service postgresql restart
```

### 3. Windows에서 접속
- Host: `localhost` (또는 WSL IP 확인: `wsl hostname -I`)
- Port: `5432`
- Database: `ai_npc_db`
- Username: `postgres`
- Password: `yourpassword`

---

## 방법 2: 기존 Windows PostgreSQL 사용 (수동 빌드)

### 필수 도구
1. Visual Studio 2022 Build Tools
2. Git for Windows
3. CMake

### 빌드 및 설치
```cmd
REM Visual Studio Developer Command Prompt 실행

git clone https://github.com/pgvector/pgvector.git
cd pgvector

REM PostgreSQL 경로 설정
set PGROOT=C:\Program Files\PostgreSQL\18

REM 빌드
nmake /F Makefile.win PG_CONFIG="%PGROOT%\bin\pg_config"

REM 설치
nmake /F Makefile.win PG_CONFIG="%PGROOT%\bin\pg_config" install
```

### 파일 복사 확인
- `vector.dll` → `C:\Program Files\PostgreSQL\18\lib\`
- `vector.control` → `C:\Program Files\PostgreSQL\18\share\extension\`
- `vector--*.sql` → `C:\Program Files\PostgreSQL\18\share\extension\`

### PostgreSQL 재시작
- Windows 서비스에서 `postgresql-x64-18` 재시작

---

## 방법 3: Docker Desktop 설치 (가장 간단)

Docker Desktop for Windows 설치 후:
```bash
docker run -d \
  --name postgres-pgvector \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=ai_npc_db \
  -p 5432:5432 \
  pgvector/pgvector:pg17
```

---

## 설치 확인

pgAdmin4 또는 psql에서:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT * FROM pg_extension WHERE extname = 'vector';
```

