# Miner Maintenance & Operations

This guide covers "Day 2" operations for Chutes miners: monitoring, troubleshooting, updating, and maintaining your mining infrastructure.

## Routine Maintenance

### 1. Updating Components

The Chutes ecosystem evolves rapidly. Keep your miner up to date to ensure compatibility and maximize rewards.

**Updating Charts:**
Use the provided Ansible playbooks to update your Helm charts. This pulls the latest miner and GPU agent images.

```bash
# From your ansible/k3s directory
ansible-playbook -i inventory.yml playbooks/deploy-charts.yml
```

**Updating OS & Drivers:**
Periodically update your base OS and NVIDIA drivers. **Caution:** Drain the node or set it to unschedulable in Kubernetes before rebooting to avoid slashing/penalties for dropping active chutes.

### 2. Cleaning Disk Space

HuggingFace models and Docker images can consume significant disk space. The `chutes-cacheclean` service usually handles this, but you can run manual cleanups if needed.

**Prune Docker Images:**

```bash
# On a GPU node
docker system prune -a -f --filter "until=24h"
```

**Clear HuggingFace Cache:**
Model weights are stored in the configured cache directory (default `/var/snap`). You can manually delete old models if space is critical, but this will force re-downloads for new deployments.

## Troubleshooting

### Common Issues

**1. Node Not Joining Cluster**

- **Check Wireguard**: Ensure `wg0` interface is up and has the correct IP.
  - `ip addr show wg0`
  - `systemctl status wg-quick@wg0`
- **Check K3s Agent**:
  - `systemctl status k3s-agent`
  - Logs: `journalctl -u k3s-agent -f`

**2. GPU Not Detected**

- **NVIDIA SMI**: Run `nvidia-smi` on the node. If it fails, reinstall drivers.
- **K8s Detection**: Check if the node advertises GPU resources:
  ```bash
  kubectl describe node <node-name> | grep nvidia.com/gpu
  ```
- **GPU Operator**: Ensure the NVIDIA GPU Operator pods are running in the `gpu-operator` namespace.

**3. "Gepetto" Not Scheduling Pods**

- **Check Logs**:
  ```bash
  kubectl logs -l app=gepetto -n chutes -f
  ```
- **Check Resources**: Ensure you have enough free CPU/RAM/GPU. Gepetto won't schedule if the cluster is full.
- **Check Taints**: Ensure nodes aren't tainted unexpectedly.

### Rebooting a Node Safely

To reboot a node without impacting your miner score significantly (by failing active requests):

1.  **Cordon the node** (stop new scheduling):
    ```bash
    kubectl cordon <node-name>
    ```
2.  **Wait for jobs to finish** (optional, but polite).
3.  **Reboot the node**.
4.  **Uncordon the node** once it's back online and `nvidia-smi` works:
    ```bash
    kubectl uncordon <node-name>
    ```

## Monitoring

### Grafana Dashboards

Your miner installation includes Grafana (default port 30080 on the control node).

- **Compute Overview**: View total GPU usage, active chutes, and potential earnings.
- **Node Health**: Monitor CPU, RAM, and Disk usage per node.
- **Network Traffic**: critical for ensuring you aren't bottlenecked on bandwidth (especially for image/video models).

### Logs

**Miner API Logs:**

```bash
kubectl logs -l app=miner-api -n chutes -f
```

**Instance Logs (Specific Chute):**
Find the pod name for a specific chute instance:

```bash
kubectl get pods -n chutes -l chute_id=<chute_id>
kubectl logs <pod_name> -n chutes -f
```

## Security Best Practices

- **Rotate Keys**: Periodically rotate your hotkey if you suspect compromise (requires re-registering or updating miner config).
- **Firewall**: Ensure only the API port (32000) and Wireguard port (51820) are exposed externally. All internal traffic should route over Wireguard (wg0).
- **SSH Access**: Disable password authentication and use SSH keys only.
