# Deploy Analytics Forge on Oracle Cloud Always Free VM

You cannot finish Oracle signup from this repo — that needs **your** Oracle account.
After the VM exists, deploy is one script + one firewall rule.

## 1. Create Always Free VM (console)

1. Sign up: https://www.oracle.com/cloud/free/
2. Menu → **Compute → Instances → Create instance**
3. Recommended free shape:
   - **Image:** Canonical Ubuntu 22.04 (or 24.04)
   - **Shape:** `VM.Standard.A1.Flex` (Ampere ARM) — e.g. **2 OCPU / 12 GB RAM**
4. Networking: assign a **public IP**
5. Add your **SSH public key**
6. Create instance → copy **Public IP**

## 2. Open port 8501 in Oracle (required)

Oracle blocks ports until you allow them:

1. Instance → **Subnet** → **Security List** (or NSG)
2. **Add Ingress Rules**
   - Source: `0.0.0.0/0`
   - Protocol: TCP
   - Destination port: **8501**
   - Also keep **22** open for SSH

## 3. SSH into the VM

```bash
ssh -i your-key.pem ubuntu@YOUR_PUBLIC_IP
```

(On some images the user is `opc` instead of `ubuntu`.)

## 4. Put the app on the VM

**Option A — GitHub (best)**

```bash
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/YOUR_USER/analytics-forge.git
cd analytics-forge
chmod +x deploy/setup-oracle.sh
./deploy/setup-oracle.sh
```

If Docker says permission denied, log out/in (or `newgrp docker`) and run:

```bash
docker compose up -d --build
```

**Option B — copy from your Windows PC (no GitHub yet)**

From PowerShell on your PC (with `scp` / OpenSSH):

```powershell
scp -r -i your-key.pem C:\Users\HP\analytics-forge ubuntu@YOUR_PUBLIC_IP:~/
```

Then on the VM:

```bash
cd ~/analytics-forge
chmod +x deploy/setup-oracle.sh
./deploy/setup-oracle.sh
```

## 5. Open the site

```
http://YOUR_PUBLIC_IP:8501
```

Create your email/password account in the app. Users + projects live in `./data` on the VM (Docker volume), so restarts keep history.

## 6. Useful commands on the VM

```bash
cd ~/analytics-forge
docker compose ps
docker compose logs -f --tail=100
docker compose restart
docker compose down
docker compose up -d --build   # after git pull
```

## 7. Optional: custom domain later

Point an A record of `yourname.in` to the public IP, then put Nginx/Caddy in front for HTTPS. Not required for first launch.

## Notes

- **Prophet:** not in the default Docker image (keeps ARM builds reliable). Install later with `requirements-optional.txt` if needed.
- **Free forever:** Keep the Always Free shape; stop paid shapes if you ever switch.
- **Billing alerts:** Turn on Oracle budget alerts anyway so a misclick never surprises you.
