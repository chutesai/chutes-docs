# Node bootstrapping

To ensure the highest probability of success, you should provision your servers with `Ubuntu 22.04`, preferably with NO nvidia driver installations if possible.

### Networking note before starting!!!

Before doing anything, you should check the IP addresses used by your server provider, and make sure you do not use an overlapping network for wireguard. By default, chutes uses 192.168.0.0/20 for this purpose, but that may conflict with some providers, e.g. Nebius through Shadeform sometimes uses 192.168.x.x network space. If the network overlaps, you will have conflicting entries in your route table and the machine may basically get bricked as a result.

It's quite trivial to use a different network for wireguard, or even just a different non-overlapping range in the 192.168.x.x space, but only if you start initially with that network. To migrate after you've already setup the miner with a different wireguard network config is a bit of effort.

To use a different range, simply update these four files:

1. `ansible/k3s/inventory.yml` your hosts will need the updated `wireguard_ip` values to match
2. `ansible/k3s/group_vars/all.yml` (or similar, depending on repo structure) usually defines the wireguard network. Check the variable `wireguard_network` or similar if exposed.

I would NOT recommend changing the wireguard network if you are already running, unless you absolutely need to. And if you do, the best bet is to actually completely wipe the node and start over.

#### external_ip

The chutes API/validator sends traffic directly to each GPU node, and does not route through the main CPU node at all. For the system to work, this means each GPU node must have a publicly routeable IP address on each GPU node that is not behind a shared IP (since it uses kubernetes nodePort services). This IP is the public IPv4, and must not be something in the private IP range like 192.168.0.0/16, 10.0.0.0/8, etc.

This public IP _must_ be dedicated, and be the same for both egress and ingress. This means, for a node to pass validation, when the validator connects to it, the IP address you advertise as a miner must match the IP address the validator sees when your node fetches a remote token, i.e. you can't use a shared IP with NAT/port-mapping if the underlying nodes route back out to the internet with some other IPs.

## 1. Install ansible (on your local system, not the miner node(s))

### Mac

If you haven't yet, setup homebrew:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install ansible:

```bash
brew install ansible
```

### Ubuntu/Ubuntu (WSL)/aptitude based systems

```bash
sudo apt -y update && sudo apt -y install ansible python3-pip
```

### CentOS/RHEL/Fedora

Install epel repo if you haven't (and it's not fedora)

```bash
sudo dnf install epel-release -y
```

Install ansible:

```bash
sudo dnf install ansible -y
```

## 2. Install ansible collections

```bash
ansible-galaxy collection install community.general
ansible-galaxy collection install kubernetes.core
```

## OPTIONAL: Performance Tweaks for Ansible

```bash
wget https://files.pythonhosted.org/packages/source/m/mitogen/mitogen-0.3.22.tar.gz
tar -xzf mitogen-0.3.22.tar.gz
```

Then in your ansible.cfg

```
[defaults]
strategy_plugins = /path/to/mitogen-0.3.22/ansible_mitogen/plugins/strategy
strategy = mitogen_linear
... leave the rest, and add this block below
[ssh_connection]
ssh_args = -o ControlMaster=auto -o ControlPersist=2m
```

## 3. Update inventory configuration

Clone the repository:

```bash
git clone https://github.com/chutesai/chutes-miner.git
cd chutes-miner/ansible/k3s
```

Using your favorite text editor (vim of course), edit `inventory.yml` to suite your needs.

For example:

```yaml
all:
  vars:
    # List of SSH public keys, e.g. cat ~/.ssh/id_rsa.pub
    ssh_keys:
      - "ssh-rsa AAAA... user@hostname"
      - "ssh-rsa BBBB... user2@hostname2"
    # The username you want to use to login to those machines (and your public key will be added to).
    user: billybob
    # The initial username to login with, for fresh nodes that may not have your username setup.
    ansible_user: ubuntu
    # The default validator each GPU worker node will be assigned to.
    validator: 5Dt7HZ7Zpw4DppPxFM7Ke3Cm7sDAWhsZXmM5ZAmE7dSVJbcQ
    # By default, no nodes are the primary (CPU node running all the apps, wireguard, etc.) Override this flag exactly once below.
    is_primary: false
    # We assume GPU is enabled on all nodes, but of course you need to disable this for the CPU nodes below.
    gpu_enabled: true
    # The port you'll be using for the registry proxy, MUST MATCH chart/values.yaml registry.service.nodePort!
    registry_port: 30500
    # SSH sometimes just hangs without this...
    ansible_ssh_common_args: "-o ControlPath=none"
    # SSH retries...
    ansible_ssh_retries: 3
    # Ubuntu major/minor version.
    ubuntu_major: "22"
    ubuntu_minor: "04"
    # CUDA version - leave as-is unless using h200s, in which case either use 12-5 or skip_cuda: true (if provider already pre-installed drivers)
    cuda_version: "12-6"
    # NVIDA GPU drivers - leave as-is unless using h200s, in which case it would be 555
    nvidia_version: "560"
    # Flag to skip the cuda install entirely, if the provider already has cuda 12.x+ installed (note some chutes will not work unless 12.6+)
    skip_cuda: false

    # PATH TO YOUR HOTKEY FILE
    # This is used to create the miner-credentials secret in k8s automatically
    hotkey_path: ~/.bittensor/wallets/default/hotkeys/my-hotkey

    # Setup local kubeconfig?
    # If true, it will copy the kubeconfig from the primary node to your local machine
    setup_local_kubeconfig: true

  hosts:
    # This would be the main node, which runs postgres, redis, gepetto, etc.
    chutes-miner-cpu-0:
      ansible_host: 1.0.0.0
      external_ip: 1.0.0.0
      wireguard_ip: 192.168.0.1
      gpu_enabled: false
      is_primary: true
      wireguard_mtu: 1420 # optional (default is 1380)

    # These are the GPU nodes, which actually run the chutes.
    chutes-miner-gpu-0:
      ansible_host: 1.0.0.1
      external_ip: 1.0.0.1
      wireguard_ip: 192.168.0.3
```

## 4. Run the playbook

This playbook handles wireguard setup, k3s installation, and joining nodes to the cluster.

```bash
ansible-playbook -i inventory.yml site.yml
```

## 5. Install 3rd party helm charts

This step will install nvidia GPU operator and prometheus on your servers.

You need to run this one time only (although running it again shouldn't cause any problems).

```bash
ansible-playbook -i inventory.yml extras.yml
```

## To add a new node, after the fact

First, update your `inventory.yml` with the new host configuration.

Then, run the site playbook with `--limit` to target only the new node (and the primary, as it's needed for coordination/token generation usually, though specific instructions may vary, running on all is safest but slower).

```bash
ansible-playbook -i inventory.yml site.yml --limit chutes-h200-0,chutes-miner-cpu-0
```

(Including the primary node ensures that if any coordination is needed, it is available).

Then run extras on the new node:

```bash
ansible-playbook -i inventory.yml extras.yml --limit chutes-h200-0
```
