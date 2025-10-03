<<<<<<< HEAD
# 🚀 Deployment Guide - PlantAI Disease Detector

This guide will help you deploy your PlantAI Disease Detector to Railway.app and other platforms.

## 📋 Prerequisites

- GitHub account
- Railway.app account
- Git installed on your system
- Python 3.8+ installed locally

## 🚀 Railway.app Deployment

### Step 1: Create GitHub Repository

1. **Create a new repository on GitHub**
   - Go to [GitHub](https://github.com)
   - Click "New repository"
   - Name it `plant-disease-detector`
   - Make it public
   - Don't initialize with README (we already have one)

2. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit: PlantAI Disease Detector"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/plant-disease-detector.git
   git push -u origin main
   ```

### Step 2: Deploy to Railway.app

1. **Sign up for Railway.app**
   - Go to [Railway.app](https://railway.app)
   - Sign up with your GitHub account

2. **Create a new project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your `plant-disease-detector` repository

3. **Configure deployment**
   - Railway will automatically detect it's a Python project
   - It will use the `nixpacks.toml` configuration
   - The app will start automatically

4. **Set environment variables** (Optional)
   - Go to your project settings
   - Add environment variables:
     ```
     FLASK_ENV=production
     SECRET_KEY=your-secure-secret-key
     ```

### Step 3: Verify Deployment

1. **Check deployment status**
   - Go to your Railway dashboard
   - Check the deployment logs
   - Ensure the app starts successfully

2. **Test your application**
   - Click on the generated URL
   - Test all features:
     - Home page loads
     - Image upload works
     - Live detection works
     - Training page loads
     - Analytics dashboard works

## 🔧 Configuration Files

### nixpacks.toml
```toml
[phases.setup]
nixPkgs = ['python39', 'pip']

[phases.install]
cmds = ['pip install -r requirements.txt']

[start]
cmd = 'gunicorn app:app'
```

### Procfile
```
web: gunicorn app:app
```

### railway.json
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "gunicorn app:app",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300
  }
}
```

## 🌐 Custom Domain (Optional)

1. **Add custom domain in Railway**
   - Go to your project settings
   - Click "Domains"
   - Add your custom domain
   - Follow the DNS configuration instructions

2. **Update DNS records**
   - Add CNAME record pointing to your Railway domain
   - Wait for DNS propagation (up to 24 hours)

## 📊 Monitoring and Analytics

### Railway Dashboard
- Monitor CPU and memory usage
- View deployment logs
- Check error rates
- Monitor response times

### Application Health
- Health check endpoint: `/health`
- Real-time analytics: `/analytics`
- Model status: `/model/info`

## 🔒 Security Considerations

### Environment Variables
- Never commit sensitive data to Git
- Use Railway's environment variables for secrets
- Rotate SECRET_KEY regularly

### Production Settings
- Debug mode is disabled in production
- Secure headers are configured
- File upload limits are enforced
- CORS is properly configured

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures**
   - Check `requirements.txt` for version conflicts
   - Ensure all dependencies are compatible
   - Check Railway build logs

2. **Runtime Errors**
   - Check application logs in Railway dashboard
   - Verify environment variables are set
   - Ensure model files are present

3. **Performance Issues**
   - Monitor resource usage in Railway dashboard
   - Consider upgrading to a higher plan
   - Optimize model loading

### Debug Commands

```bash
# Check application status
curl https://your-app.railway.app/health

# View logs
railway logs

# Connect to Railway CLI
railway login
railway link
```

## 📈 Scaling

### Horizontal Scaling
- Railway automatically handles load balancing
- Multiple instances can be deployed
- Database connections are pooled

### Vertical Scaling
- Upgrade Railway plan for more resources
- Optimize model loading and caching
- Use CDN for static assets

## 🔄 Continuous Deployment

### Automatic Deployments
- Railway automatically deploys on Git push
- Branch-based deployments supported
- Preview deployments for pull requests

### Manual Deployments
```bash
# Deploy specific branch
railway up --detach

# Deploy with specific environment
railway up --environment production
```

## 📝 Maintenance

### Regular Updates
- Keep dependencies updated
- Monitor security advisories
- Update model files as needed

### Backup Strategy
- Model files are in Git repository
- Database backups (if using external DB)
- Configuration backups

## 🆘 Support

### Railway Support
- [Railway Documentation](https://docs.railway.app)
- [Railway Discord](https://discord.gg/railway)
- [Railway GitHub](https://github.com/railwayapp)

### Application Support
- Check application logs
- Review error messages
- Test locally first

---

## 🎉 Success!

Once deployed, your PlantAI Disease Detector will be available at:
`https://your-app-name.railway.app`

Share your deployed application and start helping farmers detect plant diseases! 🌱
=======
# 🚀 Deployment Guide - PlantAI Disease Detector

This guide will help you deploy your PlantAI Disease Detector to Railway.app and other platforms.

## 📋 Prerequisites

- GitHub account
- Railway.app account
- Git installed on your system
- Python 3.8+ installed locally

## 🚀 Railway.app Deployment

### Step 1: Create GitHub Repository

1. **Create a new repository on GitHub**
   - Go to [GitHub](https://github.com)
   - Click "New repository"
   - Name it `plant-disease-detector`
   - Make it public
   - Don't initialize with README (we already have one)

2. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit: PlantAI Disease Detector"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/plant-disease-detector.git
   git push -u origin main
   ```

### Step 2: Deploy to Railway.app

1. **Sign up for Railway.app**
   - Go to [Railway.app](https://railway.app)
   - Sign up with your GitHub account

2. **Create a new project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your `plant-disease-detector` repository

3. **Configure deployment**
   - Railway will automatically detect it's a Python project
   - It will use the `nixpacks.toml` configuration
   - The app will start automatically

4. **Set environment variables** (Optional)
   - Go to your project settings
   - Add environment variables:
     ```
     FLASK_ENV=production
     SECRET_KEY=your-secure-secret-key
     ```

### Step 3: Verify Deployment

1. **Check deployment status**
   - Go to your Railway dashboard
   - Check the deployment logs
   - Ensure the app starts successfully

2. **Test your application**
   - Click on the generated URL
   - Test all features:
     - Home page loads
     - Image upload works
     - Live detection works
     - Training page loads
     - Analytics dashboard works

## 🔧 Configuration Files

### nixpacks.toml
```toml
[phases.setup]
nixPkgs = ['python39', 'pip']

[phases.install]
cmds = ['pip install -r requirements.txt']

[start]
cmd = 'gunicorn app:app'
```

### Procfile
```
web: gunicorn app:app
```

### railway.json
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "gunicorn app:app",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300
  }
}
```

## 🌐 Custom Domain (Optional)

1. **Add custom domain in Railway**
   - Go to your project settings
   - Click "Domains"
   - Add your custom domain
   - Follow the DNS configuration instructions

2. **Update DNS records**
   - Add CNAME record pointing to your Railway domain
   - Wait for DNS propagation (up to 24 hours)

## 📊 Monitoring and Analytics

### Railway Dashboard
- Monitor CPU and memory usage
- View deployment logs
- Check error rates
- Monitor response times

### Application Health
- Health check endpoint: `/health`
- Real-time analytics: `/analytics`
- Model status: `/model/info`

## 🔒 Security Considerations

### Environment Variables
- Never commit sensitive data to Git
- Use Railway's environment variables for secrets
- Rotate SECRET_KEY regularly

### Production Settings
- Debug mode is disabled in production
- Secure headers are configured
- File upload limits are enforced
- CORS is properly configured

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures**
   - Check `requirements.txt` for version conflicts
   - Ensure all dependencies are compatible
   - Check Railway build logs

2. **Runtime Errors**
   - Check application logs in Railway dashboard
   - Verify environment variables are set
   - Ensure model files are present

3. **Performance Issues**
   - Monitor resource usage in Railway dashboard
   - Consider upgrading to a higher plan
   - Optimize model loading

### Debug Commands

```bash
# Check application status
curl https://your-app.railway.app/health

# View logs
railway logs

# Connect to Railway CLI
railway login
railway link
```

## 📈 Scaling

### Horizontal Scaling
- Railway automatically handles load balancing
- Multiple instances can be deployed
- Database connections are pooled

### Vertical Scaling
- Upgrade Railway plan for more resources
- Optimize model loading and caching
- Use CDN for static assets

## 🔄 Continuous Deployment

### Automatic Deployments
- Railway automatically deploys on Git push
- Branch-based deployments supported
- Preview deployments for pull requests

### Manual Deployments
```bash
# Deploy specific branch
railway up --detach

# Deploy with specific environment
railway up --environment production
```

## 📝 Maintenance

### Regular Updates
- Keep dependencies updated
- Monitor security advisories
- Update model files as needed

### Backup Strategy
- Model files are in Git repository
- Database backups (if using external DB)
- Configuration backups

## 🆘 Support

### Railway Support
- [Railway Documentation](https://docs.railway.app)
- [Railway Discord](https://discord.gg/railway)
- [Railway GitHub](https://github.com/railwayapp)

### Application Support
- Check application logs
- Review error messages
- Test locally first

---

## 🎉 Success!

Once deployed, your PlantAI Disease Detector will be available at:
`https://your-app-name.railway.app`

Share your deployed application and start helping farmers detect plant diseases! 🌱
>>>>>>> e1fcd1d8ea3d427a90f7cd895c6c465448981fcb
