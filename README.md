# AR Earring Try-On Web App

A real-time AR earring try-on application that uses your camera to overlay virtual earrings on your face.

## Features

- Real-time face tracking using MediaPipe
- Smooth earring placement with One Euro Filter stabilization
- Automatic visibility detection (left/right ear)
- Perspective-aware scaling based on head rotation
- Graceful fade in/out animations

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:5000`

## Deploy to Render

1. Push your code to a GitHub repository

2. Go to [Render Dashboard](https://dashboard.render.com/)

3. Click "New +" and select "Web Service"

4. Connect your GitHub repository

5. Configure the service:
   - **Name**: ar-earring-tryon (or your preferred name)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free (or your preferred tier)

6. Click "Create Web Service"

7. Wait for the deployment to complete (first deploy may take 5-10 minutes)

8. Access your app at the provided Render URL

## Files

- `app.py` - Flask web application with AR processing
- `templates/index.html` - Frontend with camera access
- `earring.png` - Earring image asset
- `requirements.txt` - Python dependencies
- `Procfile` - Render deployment configuration

## Browser Requirements

- Modern browser with WebRTC support (Chrome, Firefox, Safari, Edge)
- Camera permissions must be granted
- HTTPS required for camera access (Render provides this automatically)

## Troubleshooting

- If camera doesn't work, ensure you've granted camera permissions
- If deployment fails, check Render logs for errors
- For slow processing, consider upgrading to a paid Render tier
