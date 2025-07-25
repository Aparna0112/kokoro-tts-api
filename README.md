# kokoro-tts-api
High-quality Text-to-Speech API using Kokoro-82M with JWT authentication
# Kokoro TTS API with JWT Authentication

High-quality Text-to-Speech API using Kokoro-82M model with JWT authentication, optimized for Render deployment.

## Features

- 🎤 **High-Quality TTS**: Kokoro-82M model with 21 different voices
- 🔐 **JWT Authentication**: Secure user registration and login
- 🌍 **Multiple Accents**: American and British voices
- ⚡ **Memory Optimized**: Works on Render free tier
- 🎛️ **Speed Control**: Adjustable speech speed (0.5x - 2.0x)
- 📱 **RESTful API**: Complete CRUD operations

## Available Voices

- **American Female**: Bella, Nova, Shimmer, Echo, Fable, Heart, Alloy, Aoede, Onyx
- **American Male**: Adam, Liam, Fin, Domi, Sarah
- **British Female**: Emma, Isabella, Jenny, Sky  
- **British Male**: George, Lewis, William

## API Endpoints

- `POST /auth/register` - Register new user
- `POST /auth/login` - Login and get JWT token
- `GET /voices` - List available voices
- `POST /synthesize` - Generate speech
- `GET /audio/{audio_id}` - Download audio file

## Deployment

1. Fork/clone this repository
2. Connect to Render
3. Deploy using the included `render.yaml`
4. Set environment variable `SECRET_KEY`

## Usage

```bash
# Register user
curl -X POST "https://your-app.onrender.com/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "email": "user@example.com", "password": "password123"}'

# Login
curl -X POST "https://your-app.onrender.com/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "password123"}'

# Generate speech
curl -X POST "https://your-app.onrender.com/synthesize" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"text": "Hello world!", "voice": "af_bella", "speed": 1.0}'
