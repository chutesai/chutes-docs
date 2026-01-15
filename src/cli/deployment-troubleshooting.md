# Deployment troubleshooting steps

### Chute is deployed but cold

Your chute image built and deployed successfully but is in a cold state.


**Run the warmup endpoint**

```bash
curl -X GET "https://api.chutes.ai/chutes/warmup/{chute_id_or_name}"
  -H "Content-Type: application/json"

```
This endpoint will notifiy the miners that this Chute is ready for use and will cause miners to deploy instances. Depending on the configuration of your Chute the warmup can take some time as it downloads any required files. It is possible that the message will timeout before the Chute actually goes hot so please be aware. 

**Check for active instances**

```bash
# get the Chutes info and filter out everything but the instances.
chutes chutes get your_chute_ID_here | jq '.instances'
```

**example output**

```bash
Mac:~ chutist$ chutes chutes get 8f2105c5-b200-5aa5-969f-0720f7690f3c | jq '.instances'
2026-01-14 14:43:26.979 | INFO     | chutes.config:get_config:55 - Loading chutes config from /Users/algowarry/.chutes/config.ini...
2026-01-14 14:43:26.979 | DEBUG    | chutes.config:get_config:78 - Configured chutes: with api_base_url=https://api.chutes.ai
2026-01-14 14:43:26.979 | DEBUG    | chutes.util.auth:sign_request:67 - Signing message: 5CPBAsApHkvdhviSry2xoN38MLX5xG5FUPy3QVWVPe46JxCT:1768419806:chutes
2026-01-14 14:43:27.189 | INFO     | chutes.crud:_get_object:161 - chute 8f2105c5-b200-5aa5-969f-0720f7690f3c:
[
  {
    "instance_id": "b3f083a3-3460-4d37-83e1-6c945c5d3c61",
    "region": "n/a",
    "active": true,
    "verified": true,
    "last_verified_at": "2026-01-14T19:38:24.773055Z"
  },
  {
    "instance_id": "dbf02090-aa98-4584-98c1-2f36899050b8",
    "region": "n/a",
    "active": true,
    "verified": true,
    "last_verified_at": "2026-01-14T19:38:24.773055Z"
  },
  {
    "instance_id": "e6c2eb0e-4b61-43da-94cd-e17c933e6159",
    "region": "n/a",
    "active": true,
    "verified": true,
    "last_verified_at": "2026-01-14T19:38:24.773055Z"
  }
]

```
From the listed instances look for the "active" tag. If it is set to true the Chute is hot. If all of the "active" tags are set to false those instances are still starting and have not become hot yet. "active": true is hot and "active": false is starting up. 

If you see no instances at all then the warmup command failed and the Chute is sitting cold in an unused state. 


## You ran the warmup but your Chute wont go hot

Sometimes your build and deployment will succeed but the warmup will fail. This is usually due to a configuration error or limitation within your source file. The first common places to check are as follows.

### Model revision

One of the first places to check on your source file is the model revision. This identifier is pulled from the huggingface.co repo for the model you are attempting to deploy. 

**Example source file with revision tag**

```bash
chute = build_sglang_chute(
    username="chutes",
    readme="Qwen/Qwen3-32B",
    model_name="Qwen/Qwen3-32B",
    image="chutes/sglang:nightly-2025120900",
    revision="ba1f828c09458ab0ae83d42eaacc2cf8720c7957",
    concurrency=64,
    

```

To locate this revision identifier go to the models page on huggingface.co and click the Files and versions tab. Then on the right side click the commit history button. On the commit history page you will see all commits that have been made to the model. 

Click the small copy icon next to the most recent commit and you will have the revision identifier. This is the identifier that need to be on the revision tag in your source file. 

If the revision identifier in your source file does not match one on the models commit history page the Chute will fail to go hot. If you have confirmed the revisions are correct the next place to check is the node selector.



## Node Selector

The node selector is the part of the Chutes source file that dictates the number of GPU's and GB of VRAM required to run your Chute. 

**Example correct Node Selector**


```bash
  node_selector=NodeSelector(
        gpu_count=4,
        min_vram_gb_per_gpu=80,
    ),
```

The node selector listed above is simplified and allows for a broad variety of cards to power your Chute. This is the recommended format as it allows any available nodes that fit your minimum GB of VRAM

**example potentially problematic node selector**

```bash
  node_selector=NodeSelector(
        gpu_count=1,
        include=["A100", "h100"],
```
This node selector is very specific about the types of GPU's it wants. This can cause an issue if there are a limited number of those cards in the available inventory. If a matching card is not available the chute will fail to go hot even if everything else is correct.

## Pull Logs

At this point if you have confirmed all of the above steps and your revisions and node selector are correct the next step is to review logs. To do this follow the following steps.

**Warm up the chute**

```bash
curl -X GET "https://api.chutes.ai/chutes/warmup/{chute_id_or_name}"
  -H "Content-Type: application/json"

```
now check for instances and not the instance ID's ( see above for examples)

**Check for instances**
```bash
# get the Chutes info and filter out everything but the instances.
chutes chutes get your_chute_ID_here | jq '.instances'
```
now once you have those instance ID's you can pull live logs from those instances.

**Pull logs**

```bash
curl -N -H "Authorization: Bearer $YOUR_API_KEY_HERE" 
https://api.chutes.ai/instances/INSTANCE_ID_HERE/logs
```
You will not get live logs from the instance as it is preparing your Chute. If the instance fails the logs will end and you will see an error that caused the instance to fail. Use that to troubleshoot your Chute and resolve the issue. Ff needed create a ticket either on the site or Discord and present those logs for one of our team to review. 



### Support Resources

- 📖 **Documentation**: [Complete Docs](/docs)
- 💬 **Discord**: [Community Chat](https://discord.gg/wHrXwWkCRz)
- 📨 **Support**: [Email](support@chutes.ai)
- 🐛 **Issues**: [GitHub Issues](https://github.com/chutesai/chutes/issues)

---

Continue to specific command documentation:

- **[Account Management](/docs/cli/account)** - Detailed account commands
- **[Building Images](/docs/cli/build)** - Advanced build options
- **[Deploying Chutes](/docs/cli/deploy)** - Deployment strategies
- **[Managing Resources](/docs/cli/manage)** - Resource management
