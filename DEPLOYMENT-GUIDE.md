# ☁️ Streamlit Cloud Deployment Guide

## Deploy QuantumSignal Labs to Streamlit Cloud

---

## 🚀 Quick Deployment (5 Minutes)

### Step 1: Prepare GitHub Repository

1. **Upload files to your GitHub repository:**
   - `ai_trading_dashboard.py`
   - `requirements.txt`
   - `README-PHASE2.md`

2. **Ensure repository is public**

3. **Commit and push all changes**

### Step 2: Deploy on Streamlit Cloud

1. **Go to** https://share.streamlit.io/

2. **Sign in with GitHub**

3. **Click "New app"**

4. **Configure deployment:**
   - Repository: `yogi2022/The_Dev_Premier_League_QuantumSignal_Labs`
   - Branch: `main` (or your branch name)
   - Main file path: `ai_trading_dashboard.py`

5. **Click "Deploy!"**

6. **Wait 2-3 minutes for deployment**

7. **Your app will be live at:**
   `https://share.streamlit.io/yogi2022/the-dev-premier-league-quantumsignal-labs/main/ai_trading_dashboard.py`

---

## 🔧 Configuration

### No Environment Variables Needed!
The API keys are hardcoded in the application for easy demo:
- Alpha Vantage: `2KNU40IVNB9EONNK`
- News API: `fc9118ab4dab4d46817a011698909393`

### Automatic Features:
- ✅ Dependencies auto-installed from `requirements.txt`
- ✅ App auto-starts on deployment
- ✅ HTTPS enabled by default
- ✅ Auto-scaling included

---

## 📱 Share Your Demo

Once deployed, you'll get a shareable URL like:
```
https://share.streamlit.io/[username]/[repo]/[branch]/[file]
```

**Use this URL for:**
- Phase 2 submission form
- Presentation deck
- Demo video description
- README documentation

---

## 🛠️ Alternative: Deploy on Your Own Domain

### Option 1: Heroku

**Create `Procfile`:**
```
web: streamlit run ai_trading_dashboard.py --server.port=$PORT --server.address=0.0.0.0
```

**Create `setup.sh`:**
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

**Deploy:**
```bash
heroku create quantumsignal-labs
git push heroku main
```

### Option 2: AWS EC2

**Launch EC2 Instance:**
```bash
# Install Python
sudo apt update
sudo apt install python3-pip

# Clone repo
git clone [your-repo-url]
cd [repo-name]

# Install dependencies
pip3 install -r requirements.txt

# Run app
streamlit run ai_trading_dashboard.py --server.port=80
```

### Option 3: Docker

**Create `Dockerfile`:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ai_trading_dashboard.py .

EXPOSE 8501

CMD ["streamlit", "run", "ai_trading_dashboard.py"]
```

**Build and run:**
```bash
docker build -t quantumsignal-labs .
docker run -p 8501:8501 quantumsignal-labs
```

---

## 🎯 Best Practices for Demo

### 1. Test Before Sharing
- Visit your deployed URL
- Test all features
- Verify data fetching works
- Check all tabs load correctly

### 2. Prepare Demo Data
- Use recommended stocks: AAPL, MSFT, GOOGL
- Fetch data before presentation
- Have screenshots as backup

### 3. Monitor Performance
- Check Streamlit Cloud logs
- Watch for API rate limits
- Monitor response times

### 4. Optimize for Judges
- Add clear instructions in the app
- Include tooltips and help text
- Ensure professional appearance
- Test on different browsers

---

## 📊 Monitoring & Analytics

### Streamlit Cloud Dashboard
- View app analytics
- Monitor usage
- Check error logs
- Track performance

### Custom Analytics (Optional)
Add Google Analytics to track:
- Visitor count
- Feature usage
- Session duration
- User engagement

---

## 🔒 Security Notes

### API Key Security
**Current setup:** Keys are hardcoded for demo purposes.

**For production:**
1. Use Streamlit Secrets:
   ```python
   import streamlit as st
   av_key = st.secrets["ALPHA_VANTAGE_KEY"]
   news_key = st.secrets["NEWS_API_KEY"]
   ```

2. Add secrets in Streamlit Cloud dashboard

3. Never commit secrets to GitHub

---

## 🐛 Troubleshooting

### App Won't Start
- Check `requirements.txt` syntax
- Verify file names match exactly
- Review Streamlit Cloud logs
- Ensure Python 3.8+ compatibility

### Slow Performance
- Enable caching (already implemented)
- Reduce API calls
- Optimize data processing
- Use Streamlit's `@st.cache_data`

### API Errors
- Check rate limits
- Verify API keys are valid
- Test API endpoints separately
- Add retry logic

---

## 📈 Performance Optimization

### Already Implemented:
- ✅ Streamlit caching (`@st.cache_data`)
- ✅ Rate limiting for APIs
- ✅ Efficient data processing
- ✅ Lazy loading of charts

### Additional Optimizations:
- Use Streamlit's session state
- Minimize widget redraws
- Compress images
- Enable Streamlit Cloud's CDN

---

## 🎊 Going Live Checklist

- [ ] Code pushed to GitHub
- [ ] Repository is public
- [ ] All files present (app, requirements, README)
- [ ] Streamlit Cloud account created
- [ ] App deployed successfully
- [ ] All features tested
- [ ] URL is accessible
- [ ] Demo data prepared
- [ ] Screenshots captured
- [ ] URL added to submission

---

## 📞 Support Resources

**Streamlit Documentation:**
- https://docs.streamlit.io/
- https://docs.streamlit.io/streamlit-cloud

**Streamlit Community:**
- https://discuss.streamlit.io/
- Discord: https://discord.gg/streamlit

**Repository Issues:**
- GitHub Issues for bug reports
- Discussions for questions

---

## 🌟 Pro Tips

### 1. Custom Domain (Optional)
Link your own domain to Streamlit Cloud:
- Configure CNAME record
- Update DNS settings
- Add SSL certificate

### 2. Branding
Customize app appearance:
```python
st.set_page_config(
    page_title="QuantumSignal Labs",
    page_icon="📈",
    menu_items={
        'Get Help': 'https://github.com/yogi2022/...',
        'Report a bug': 'https://github.com/yogi2022/.../issues',
        'About': 'QuantumSignal Labs - AI Trading Platform'
    }
)
```

### 3. Analytics Integration
Add tracking to understand usage:
```python
# Google Analytics
st.components.v1.html("""
<script async src="https://www.googletagmanager.com/gtag/js?id=YOUR-GA-ID"></script>
""", height=0)
```

---

## 🚀 Ready to Deploy!

Your app is now ready to share with the world!

**Demo URL Format:**
```
https://share.streamlit.io/yogi2022/the-dev-premier-league-quantumsignal-labs/main/ai_trading_dashboard.py
```

**Remember to:**
1. Test thoroughly before submission
2. Share URL in all submission materials
3. Monitor performance during evaluation
4. Be ready for live demo

---

**Good luck with your deployment!** 🎉

*Happy Trading!* 📈
