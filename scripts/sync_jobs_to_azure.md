# Getting processed jobs onto the Azure VM

1. Download `jobs.zip` from the Kaggle notebook's Output tab.
2. Copy and unzip on the server (replace user/host/path):

```bash
scp jobs.zip azureuser@<VM_IP>:~/app/backend/
ssh azureuser@<VM_IP> "cd ~/app/backend && unzip -o jobs.zip -d . && rm jobs.zip"
```

`jobs.zip` contains a top-level `jobs/` folder, so it unpacks straight into
`backend/jobs/`. No server restart is needed: job folders are read on request.

Tips
- Open port 8000 (or put the app behind nginx on 80/443) in the VM's network rules.
- Set `CORS_ORIGINS` in `.env` to your frontend URL.
- Stop/deallocate the VM when you are not testing to save credit, and set a
  budget alert in Azure Cost Management.
- Docker option: `cd backend && docker build -t vke-api . && docker run -p 8000:8000 -v $PWD/jobs:/app/jobs --env-file ../.env vke-api`
