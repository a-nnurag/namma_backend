# NammaKelsa — Backend API

FastAPI backend for the NammaKelsa skill verification platform. Handles citizen and admin authentication, job applications, interview sessions, and coordinates with the ML service for AI-powered candidate evaluation.

## Tech Stack

- **Python 3.11** + **FastAPI** (async)
- **PostgreSQL 16** with **pgvector** (face embedding uniqueness at registration)
- **Redis** (OTP, JWT session cache, cooldown cache, ML status poll cache, chunked upload state)
- **Kafka / Redpanda** (video/audio chunk streaming to ML service)
- **SQLAlchemy** (async ORM via asyncpg)
- **JWT** (HS256, httpOnly cookie via Next.js BFF)

## API Surface

| Group | Prefix | Description |
|-------|--------|-------------|
| Citizen Auth | `/auth` | OTP send/verify, JWT issue |
| Admin Auth | `/admin/auth` | Officer/super_admin OTP send/verify |
| Registration | `/registration` | Face + Aadhaar verification flow |
| Skills | `/skills` | Skill catalog |
| Applications | `/application` | Apply, track status, ML result polling |
| Interview | `/interview` | Session management, video/audio chunk upload via Kafka |
| Documents | `/documents` | Work experience video, degree upload |
| Admin Ops | `/admin` | User management, FRAUD flagging, stats |

## Related Repositories

| Service | Repository |
|---------|-----------|
| Frontend (Next.js) | https://github.com/a-nnurag/namma_frontend |
| ML Service | https://github.com/a-nnurag/namma_mlservice |
| Infrastructure (Docker + DB init) | https://github.com/a-nnurag/namma_infra |

## Local Development

```bash
# 1. Spin up dependencies (Postgres, Redis, Redpanda)
cd ../  # go to infra root
docker compose up postgres-backend redis redpanda -d

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your values

# 5. Run
python main.py
# API available at http://localhost:8001
# Docs at http://localhost:8001/docs
```

## Environment Variables

Copy `.env.example` to `.env` and fill in values. See `planning/BACKEND_PLAN.txt` for full configuration reference.

## Architecture Notes

- All auth tokens are issued as JWTs and stored in **httpOnly cookies** by the Next.js BFF — the token never reaches browser JavaScript.
- Redis DB 2 is used exclusively by this service. See `planning/REDIS_DESIGN.txt` for the full key schema.
- The backend **never calls the ML service directly** for status — it polls a Redis cache that a background task keeps fresh.
