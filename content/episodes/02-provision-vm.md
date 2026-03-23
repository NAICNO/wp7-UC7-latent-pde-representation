# VM Provisioning

```{objectives}
- Create a MyAccessID account using your Feide credentials
- Provision a GPU-enabled virtual machine on the NAIC Orchestrator
- Download and secure your SSH key for VM access
- Verify connectivity to the provisioned VM
```

```{admonition} Who needs this step?
:class: note

This episode is for participants from Norwegian academic institutions who will use the NAIC self-service portal to provision a virtual machine. If you already have access to a machine with Python 3.8+ and a GPU (optional), you can skip directly to [Episode 03: Setting Up the Environment](03-setup-environment.md).

If you do not have Feide access, contact **support@naic.no** to request a VM allocation.
```

## MyAccessID

To access [orchestrator.naic.no](https://orchestrator.naic.no), you need a MyAccessID linked to your institutional Feide account.

### Supported Institutions

<p><label for="institution">Check if your institution is registered:</label></p>
<select id="institution" name="institution">
  <option value="" disabled selected> MyAccessID registered Institutes </option>
  <option>University of Oslo</option>
  <option>UiT The Arctic University of Norway</option>
  <option>University of Agder</option>
  <option>University of Bergen</option>
  <option>University of Stavanger</option>
  <option>NTNU</option>
  <option>Meteorologisk institutt</option>
  <option>NILU</option>
  <option>NMBU Norwegian University of Life Sciences</option>
  <option>NORCE Norwegian Research Center</option>
  <option>Norsk Regnesentral</option>
  <option>OsloMet -- Oslo Metropolitan University</option>
  <option>Sikt - Kunnskapssektorens tjenesteleverandor</option>
  <option>Simula</option>
  <option>SINTEF</option>
  <option>Veterinaeinstituttet</option>
</select>
<p></p>

```{warning}
If your institution is not listed, it has not been registered with MyAccessID and your Feide account will not work with the NAIC Orchestrator. Contact your institution's IT department to request registration. Alternatively, contact **support@naic.no** for manual VM provisioning.
```

### Registration

If your institution is listed but you have not used MyAccessID before, register at:

https://puhuri.neic.no/user_guides/myaccessid_registration/

Registration typically takes a few minutes. You will need your institutional Feide credentials.

## Create a VM Using NAIC Orchestrator

Ensure your MyAccessID is active before proceeding. If you could not get it working, contact the workshop organizers for an alternative solution.

### Step 1: Log In

Navigate to [https://orchestrator.naic.no](https://orchestrator.naic.no) and log in with your MyAccessID credentials.

<img src="images/orchestrator1.png" alt="Orchestrator login page" width="40%" />

### Step 2: Navigate to VM Creation

Click the "Create" button in the Orchestrator dashboard.

<img src="images/orchestrator2.png" alt="Navigate to create VM" width="40%" />

### Step 3: Name Your VM and Generate SSH Key

- Provide a simple name using only alphabetical characters (no spaces)
- Create a named SSH key and **download it immediately**

```{warning}
The Orchestrator does **not** store your SSH key. If you lose the downloaded key file, you will lose access to the VM and must provision a new one.
```

<img src="images/orchestrator3.png" alt="Name VM and download SSH key" width="40%" />

### Step 4: Configure Network Access

- The Orchestrator auto-detects your current IP address
- You can optionally add your university network IP range
- SSH access is restricted to the whitelisted IPs only

<img src="images/orchestrator4.png" alt="Configure IP whitelist" width="40%" />

```{admonition} Working from multiple locations?
:class: tip

If you plan to access the VM from different networks (home, office, VPN), add all relevant IP addresses during provisioning. You can update the whitelist later through the Orchestrator dashboard.
```

### Step 5: Wait for Provisioning

The VM provisioning typically takes 2-5 minutes. The dashboard will show a progress indicator.

### Step 6: Review Connection Details

Once provisioning completes, the Orchestrator displays a customized help page with your VM's IP address and connection instructions.

### Step 7: Connect to the VM

```bash
# Set correct permissions on your SSH key
chmod 600 /path/to/your-key.pem

# Connect to the VM
ssh -i /path/to/your-key.pem ubuntu@<VM_IP_ADDRESS>
```

## Verify Your VM

After connecting, run a quick check:

```bash
# Check Python availability
python3 --version

# Check GPU (if provisioned with GPU)
nvidia-smi

# Check available disk space
df -h /home
```

```{admonition} VM Lifetime
:class: warning

NAIC Orchestrator VMs are **short-lived** and intended for workshop or development use. Save any important results to your local machine before the VM expires. Use `rsync` or `scp` to transfer files:

    scp -i /path/to/your-key.pem ubuntu@<VM_IP>:~/latent-representation-of-pde-solutions/results/* ./local-results/
```

## Troubleshooting VM Access

| Issue | Solution |
|-------|----------|
| Cannot log in to Orchestrator | Verify MyAccessID registration at puhuri.neic.no |
| SSH connection refused | Verify VM is running; check IP whitelist |
| SSH permission denied | `chmod 600 /path/to/your-key.pem` |
| SSH host key changed | `ssh-keygen -R <VM_IP>` (VM was reprovisioned) |
| Lost SSH key | Provision a new VM through the Orchestrator |
| No GPU available | VMs are allocated based on capacity; contact support@naic.no |

```{keypoints}
- MyAccessID via Feide is required for NAIC Orchestrator access
- Download and secure your SSH key immediately -- it cannot be recovered
- Your IP address must be whitelisted to connect via SSH
- VMs are short-lived; transfer results to your local machine before expiry
- Contact support@naic.no if you cannot access the Orchestrator
```
