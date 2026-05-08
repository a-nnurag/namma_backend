from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    BACKEND_PORT: int = 8001
    LOG_LEVEL: str = "INFO"

    # PostgreSQL
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/nammakelsa_backend"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_DB: int = 2

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_ADAPTER_BACKEND: str = "aiokafka"  # aiokafka | mock
    KAFKA_USE_SSL: bool = False
    KAFKA_SASL_USERNAME: str = ""
    KAFKA_SASL_PASSWORD: str = ""

    # Twilio
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_FROM_NUMBER: str = "+1XXXXXXXXXX"

    # Aadhaar verification strategy
    AADHAAR_VERIFIER_BACKEND: str = "mock"  # mock | surepass | signzy | uidai_sandbox
    SUREPASS_API_KEY: str = ""
    SUREPASS_API_URL: str = "https://kyc-api.surepass.io/api/v1/aadhaar-v2/generate-otp"
    SIGNZY_API_KEY: str = ""
    SIGNZY_API_URL: str = ""

    # Face hash strategy
    FACE_HASH_STRATEGY: str = "sha256"  # sha256 | blake2 | md5
    FACE_SIMILARITY_THRESHOLD: float = 0.85

    # Workexp video
    MAX_WORKEXP_VIDEO_SIZE_MB: int = 100
    WORKEXP_CHUNK_SIZE_BYTES: int = 65536  # 64 KB per chunk from frontend
    KAFKA_CHUNK_SIZE_BYTES: int = 262144  # 256 KB per Kafka chunk

    # JWT
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24

    # ML Service
    ML_SERVICE_URL: str = "http://ml-service:8000"

    # SarvamAI
    SARVAM_API_KEY: str = ""

    # Groq
    GROQ_API_KEY: str = ""

    # File storage
    LOCAL_STORAGE_DIR: str = "/tmp/backend"

    # Polling
    ML_POLL_INTERVAL_SECONDS: int = 30

    # Face detection
    FACE_DETECTION_CONFIDENCE: float = 0.9
    FACE_VIDEO_MAX_SIZE_MB: int = 10

    # CORS — comma-separated list of allowed origins
    ALLOWED_ORIGINS: str = "http://localhost:3000"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
