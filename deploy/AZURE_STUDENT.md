# Deploy Analytics Forge on Azure (Student)

Use this while Oracle is stuck. Same Docker app — you can keep updating after deploy.

## 0. Get free student credits

1. Open https://education.github.com/pack  
2. Verify as a student (college email / student ID)  
3. Activate **Azure for Students** (~$100 credit, no credit card on many offers)  
4. Sign in at https://portal.azure.com with that student subscription  

If GitHub Education is pending, wait for approval before creating a paid-looking VM.

## 1. Create a Linux VM (portal)

1. Portal → **Virtual machines** → **Create** → **Azure virtual machine**  
2. Basics:
   - **Subscription:** Azure for Students  
   - **Resource group:** create `rg-analytics-forge`  
   - **Name:** `analytics-forge`  
   - **Region:** Central India / South India / nearest  
   - **Image:** **Ubuntu Server 22.04 LTS**  
   - **Size:** start with **B2s** (2 vCPU / 4 GB) or **B2ms** (2 vCPU / 8 GB) if credits allow  
   - **Authentication:** SSH public key  
   - Username: `azureuser`  
3. Networking:
   - Public IP: **Yes**  
   - Inbound ports: allow **SSH (22)** for now (we add 8501 next)  
4. Review + Create → download / save your private key  
5. Copy the VM **Public IP**

## 2. Open port 8501 (required)

1. VM → **Networking** → **Network settings** / NSG  
2. **Create port rule** (Inbound):
   - Destination port: **8501**  
   - Protocol: TCP  
   - Source: Any  
   - Name: `AllowStreamlit8501`

## 3. First deploy (SSH)

From PowerShell on your PC:

```powershell
ssh -i C:\path\to\your-azure-key.pem azureuser@YOUR_PUBLIC_IP
```

On the VM:

```bash
curl -fsSL https://raw.githubusercontent.com/shaikmohammedshoaib666/analytics-forge/main/deploy/setup-vm.sh -o setup-vm.sh
chmod +x setup-vm.sh
./setup-vm.sh
```

Or clone then run:

```bash
git clone https://github.com/shaikmohammedshoaib666/analytics-forge.git
cd analytics-forge
chmod +x deploy/setup-vm.sh
./deploy/setup-vm.sh
```

Open: `http://YOUR_PUBLIC_IP:8501`  
Create your email/password account in the app.

## 4. Make updates after deploy (your normal loop)

On your PC you change code → push GitHub → on Azure VM:

```bash
cd ~/analytics-forge
git pull
docker compose up -d --build
```

That rebuilds with new libs/features. Login DB stays in `./data` on the VM disk.

## 5. Useful commands

```bash
cd ~/analytics-forge
docker compose ps
docker compose logs -f --tail=100
docker compose restart
```

## Cost tips (student credits)

- Prefer **B2s / B2ms** — enough for 1 user + ML Studio + Prophet  
- **Stop the VM** in Azure Portal when you are not coding for days (saves credits)  
- Don’t pick GPU sizes  
- Watch remaining credit in Cost Management  

## Later

Final year → move same Docker setup to AWS if you want. Azure student is the bridge while Oracle is blocked.
