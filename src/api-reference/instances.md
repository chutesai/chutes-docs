# Instances API Reference

This section covers all endpoints related to instances.


## Get Instance Reconciliation Csv

Get all instance audit instance_id, deleted_at records to help reconcile audit data.


<div class="api-test-widget" data-widget-id="widget_get__instances_reconciliation_csv"></div>
<script type="application/json" data-widget-config="widget_get__instances_reconciliation_csv">{"endpoint":"/instances/reconciliation_csv","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /instances/reconciliation_csv`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Instance Compute History Csv

Get instance_compute_history records for the scoring period (last 7 days + buffer).
Used by the auditor to reconcile compute history data on startup.


<div class="api-test-widget" data-widget-id="widget_get__instances_compute_history_csv"></div>
<script type="application/json" data-widget-config="widget_get__instances_compute_history_csv">{"endpoint":"/instances/compute_history_csv","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /instances/compute_history_csv`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Launch Config


<div class="api-test-widget" data-widget-id="widget_get__instances_launch_config"></div>
<script type="application/json" data-widget-config="widget_get__instances_launch_config">{"endpoint":"/instances/launch_config","method":"GET","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"query"},{"name":"server_id","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"job_id","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /instances/launch_config`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
| server_id | string \| null | No |  |
| job_id | string \| null | No |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Get Rint Nonce

Get runtime integrity nonce for a launch config.

This endpoint consumes the nonce from Redis (one-time use).
Only available for chutes_version >= 0.4.9.


<div class="api-test-widget" data-widget-id="widget_get__instances_launch_config__config_id__nonce"></div>
<script type="application/json" data-widget-config="widget_get__instances_launch_config__config_id__nonce">{"endpoint":"/instances/launch_config/{config_id}/nonce","method":"GET","parameters":[{"name":"config_id","type":"string","required":true,"description":"","in":"path"},{"name":"Authorization","type":"string","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /instances/launch_config/{config_id}/nonce`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| config_id | string | Yes |  |
| Authorization | string | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Claim Tee Launch Config

Claim a TEE launch config, verify attestation, and receive symmetric key.


<div class="api-test-widget" data-widget-id="widget_post__instances_launch_config__config_id__tee"></div>
<script type="application/json" data-widget-config="widget_post__instances_launch_config__config_id__tee">{"endpoint":"/instances/launch_config/{config_id}/tee","method":"POST","parameters":[{"name":"config_id","type":"string","required":true,"description":"","in":"path"},{"name":"Authorization","type":"string","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"gpus":{"items":{"additionalProperties":true,"type":"object"},"type":"array","title":"Gpus"},"host":{"type":"string","title":"Host"},"port_mappings":{"items":{"$ref":"#/components/schemas/PortMap"},"type":"array","title":"Port Mappings"},"fsv":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Fsv"},"egress":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Egress"},"lock_modules":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Lock Modules"},"netnanny_hash":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Netnanny Hash"},"run_path":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Run Path"},"py_dirs":{"anyOf":[{"items":{"type":"string"},"type":"array"},{"type":"null"}],"title":"Py Dirs"},"rint_commitment":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Rint Commitment"},"rint_nonce":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Rint Nonce"},"rint_pubkey":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Rint Pubkey"},"tls_cert":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Cert"},"tls_cert_sig":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Cert Sig"},"tls_ca_cert":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Ca Cert"},"tls_client_cert":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Client Cert"},"tls_client_key":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Client Key"},"tls_client_key_password":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Client Key Password"},"e2e_pubkey":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"E2E Pubkey"},"cllmv_session_init":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Cllmv Session Init"},"env":{"type":"string","title":"Env"},"code":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Code"},"run_code":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Run Code"},"inspecto":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Inspecto"},"deployment_id":{"type":"string","title":"Deployment Id"}},"required":["gpus","host","port_mappings","env","deployment_id"]}}</script>

**Endpoint:** `POST /instances/launch_config/{config_id}/tee`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| config_id | string | Yes |  |
| Authorization | string | No |  |
| X-Chutes-Nonce | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| gpus | object[] | Yes |  |
| host | string | Yes |  |
| port_mappings | PortMap[] | Yes |  |
| fsv | string \| null | No |  |
| egress | boolean \| null | No |  |
| lock_modules | boolean \| null | No |  |
| netnanny_hash | string \| null | No |  |
| run_path | string \| null | No |  |
| py_dirs | string[] \| null | No |  |
| rint_commitment | string \| null | No |  |
| rint_nonce | string \| null | No |  |
| rint_pubkey | string \| null | No |  |
| tls_cert | string \| null | No |  |
| tls_cert_sig | string \| null | No |  |
| tls_ca_cert | string \| null | No |  |
| tls_client_cert | string \| null | No |  |
| tls_client_key | string \| null | No |  |
| tls_client_key_password | string \| null | No |  |
| e2e_pubkey | string \| null | No |  |
| cllmv_session_init | string \| null | No |  |
| env | string | Yes |  |
| code | string \| null | No |  |
| run_code | string \| null | No |  |
| inspecto | string \| null | No |  |
| deployment_id | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Verify Tee Launch Config Instance

Verify TEE launch config instance by validating symmetric key usage via dummy ports.


<div class="api-test-widget" data-widget-id="widget_put__instances_launch_config__config_id__tee"></div>
<script type="application/json" data-widget-config="widget_put__instances_launch_config__config_id__tee">{"endpoint":"/instances/launch_config/{config_id}/tee","method":"PUT","parameters":[{"name":"config_id","type":"string","required":true,"description":"","in":"path"},{"name":"Authorization","type":"string","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `PUT /instances/launch_config/{config_id}/tee`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| config_id | string | Yes |  |
| Authorization | string | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Validate Tee Launch Config Instance


<div class="api-test-widget" data-widget-id="widget_post__instances_launch_config__config_id__attest"></div>
<script type="application/json" data-widget-config="widget_post__instances_launch_config__config_id__attest">{"endpoint":"/instances/launch_config/{config_id}/attest","method":"POST","parameters":[{"name":"config_id","type":"string","required":true,"description":"","in":"path"},{"name":"Authorization","type":"string","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"gpus":{"items":{"additionalProperties":true,"type":"object"},"type":"array","title":"Gpus"},"host":{"type":"string","title":"Host"},"port_mappings":{"items":{"$ref":"#/components/schemas/PortMap"},"type":"array","title":"Port Mappings"},"fsv":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Fsv"},"egress":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Egress"},"lock_modules":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Lock Modules"},"netnanny_hash":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Netnanny Hash"},"run_path":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Run Path"},"py_dirs":{"anyOf":[{"items":{"type":"string"},"type":"array"},{"type":"null"}],"title":"Py Dirs"},"rint_commitment":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Rint Commitment"},"rint_nonce":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Rint Nonce"},"rint_pubkey":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Rint Pubkey"},"tls_cert":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Cert"},"tls_cert_sig":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Cert Sig"},"tls_ca_cert":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Ca Cert"},"tls_client_cert":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Client Cert"},"tls_client_key":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Client Key"},"tls_client_key_password":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Client Key Password"},"e2e_pubkey":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"E2E Pubkey"},"cllmv_session_init":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Cllmv Session Init"},"env":{"type":"string","title":"Env"},"code":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Code"},"run_code":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Run Code"},"inspecto":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Inspecto"},"gpu_evidence":{"items":{"additionalProperties":true,"type":"object"},"type":"array","title":"Gpu Evidence"}},"required":["gpus","host","port_mappings","env","gpu_evidence"]}}</script>

**Endpoint:** `POST /instances/launch_config/{config_id}/attest`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| config_id | string | Yes |  |
| Authorization | string | No |  |
| X-Chutes-Nonce | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| gpus | object[] | Yes |  |
| host | string | Yes |  |
| port_mappings | PortMap[] | Yes |  |
| fsv | string \| null | No |  |
| egress | boolean \| null | No |  |
| lock_modules | boolean \| null | No |  |
| netnanny_hash | string \| null | No |  |
| run_path | string \| null | No |  |
| py_dirs | string[] \| null | No |  |
| rint_commitment | string \| null | No |  |
| rint_nonce | string \| null | No |  |
| rint_pubkey | string \| null | No |  |
| tls_cert | string \| null | No |  |
| tls_cert_sig | string \| null | No |  |
| tls_ca_cert | string \| null | No |  |
| tls_client_cert | string \| null | No |  |
| tls_client_key | string \| null | No |  |
| tls_client_key_password | string \| null | No |  |
| e2e_pubkey | string \| null | No |  |
| cllmv_session_init | string \| null | No |  |
| env | string | Yes |  |
| code | string \| null | No |  |
| run_code | string \| null | No |  |
| inspecto | string \| null | No |  |
| gpu_evidence | object[] | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Claim Launch Config


<div class="api-test-widget" data-widget-id="widget_post__instances_launch_config__config_id_"></div>
<script type="application/json" data-widget-config="widget_post__instances_launch_config__config_id_">{"endpoint":"/instances/launch_config/{config_id}","method":"POST","parameters":[{"name":"config_id","type":"string","required":true,"description":"","in":"path"},{"name":"Authorization","type":"string","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"gpus":{"items":{"additionalProperties":true,"type":"object"},"type":"array","title":"Gpus"},"host":{"type":"string","title":"Host"},"port_mappings":{"items":{"$ref":"#/components/schemas/PortMap"},"type":"array","title":"Port Mappings"},"fsv":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Fsv"},"egress":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Egress"},"lock_modules":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Lock Modules"},"netnanny_hash":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Netnanny Hash"},"run_path":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Run Path"},"py_dirs":{"anyOf":[{"items":{"type":"string"},"type":"array"},{"type":"null"}],"title":"Py Dirs"},"rint_commitment":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Rint Commitment"},"rint_nonce":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Rint Nonce"},"rint_pubkey":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Rint Pubkey"},"tls_cert":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Cert"},"tls_cert_sig":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Cert Sig"},"tls_ca_cert":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Ca Cert"},"tls_client_cert":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Client Cert"},"tls_client_key":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Client Key"},"tls_client_key_password":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Client Key Password"},"e2e_pubkey":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"E2E Pubkey"},"cllmv_session_init":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Cllmv Session Init"},"env":{"type":"string","title":"Env"},"code":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Code"},"run_code":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Run Code"},"inspecto":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Inspecto"}},"required":["gpus","host","port_mappings","env"]}}</script>

**Endpoint:** `POST /instances/launch_config/{config_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| config_id | string | Yes |  |
| Authorization | string | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| gpus | object[] | Yes |  |
| host | string | Yes |  |
| port_mappings | PortMap[] | Yes |  |
| fsv | string \| null | No |  |
| egress | boolean \| null | No |  |
| lock_modules | boolean \| null | No |  |
| netnanny_hash | string \| null | No |  |
| run_path | string \| null | No |  |
| py_dirs | string[] \| null | No |  |
| rint_commitment | string \| null | No |  |
| rint_nonce | string \| null | No |  |
| rint_pubkey | string \| null | No |  |
| tls_cert | string \| null | No |  |
| tls_cert_sig | string \| null | No |  |
| tls_ca_cert | string \| null | No |  |
| tls_client_cert | string \| null | No |  |
| tls_client_key | string \| null | No |  |
| tls_client_key_password | string \| null | No |  |
| e2e_pubkey | string \| null | No |  |
| cllmv_session_init | string \| null | No |  |
| env | string | Yes |  |
| code | string \| null | No |  |
| run_code | string \| null | No |  |
| inspecto | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Verify Launch Config Instance


<div class="api-test-widget" data-widget-id="widget_put__instances_launch_config__config_id_"></div>
<script type="application/json" data-widget-config="widget_put__instances_launch_config__config_id_">{"endpoint":"/instances/launch_config/{config_id}","method":"PUT","parameters":[{"name":"config_id","type":"string","required":true,"description":"","in":"path"},{"name":"Authorization","type":"string","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `PUT /instances/launch_config/{config_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| config_id | string | Yes |  |
| Authorization | string | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Claim Graval Launch Config

Claim a Graval launch config and receive PoVW challenge.


<div class="api-test-widget" data-widget-id="widget_post__instances_launch_config__config_id__graval"></div>
<script type="application/json" data-widget-config="widget_post__instances_launch_config__config_id__graval">{"endpoint":"/instances/launch_config/{config_id}/graval","method":"POST","parameters":[{"name":"config_id","type":"string","required":true,"description":"","in":"path"},{"name":"Authorization","type":"string","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"gpus":{"items":{"additionalProperties":true,"type":"object"},"type":"array","title":"Gpus"},"host":{"type":"string","title":"Host"},"port_mappings":{"items":{"$ref":"#/components/schemas/PortMap"},"type":"array","title":"Port Mappings"},"fsv":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Fsv"},"egress":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Egress"},"lock_modules":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Lock Modules"},"netnanny_hash":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Netnanny Hash"},"run_path":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Run Path"},"py_dirs":{"anyOf":[{"items":{"type":"string"},"type":"array"},{"type":"null"}],"title":"Py Dirs"},"rint_commitment":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Rint Commitment"},"rint_nonce":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Rint Nonce"},"rint_pubkey":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Rint Pubkey"},"tls_cert":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Cert"},"tls_cert_sig":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Cert Sig"},"tls_ca_cert":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Ca Cert"},"tls_client_cert":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Client Cert"},"tls_client_key":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Client Key"},"tls_client_key_password":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tls Client Key Password"},"e2e_pubkey":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"E2E Pubkey"},"cllmv_session_init":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Cllmv Session Init"},"env":{"type":"string","title":"Env"},"code":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Code"},"run_code":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Run Code"},"inspecto":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Inspecto"}},"required":["gpus","host","port_mappings","env"]}}</script>

**Endpoint:** `POST /instances/launch_config/{config_id}/graval`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| config_id | string | Yes |  |
| Authorization | string | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| gpus | object[] | Yes |  |
| host | string | Yes |  |
| port_mappings | PortMap[] | Yes |  |
| fsv | string \| null | No |  |
| egress | boolean \| null | No |  |
| lock_modules | boolean \| null | No |  |
| netnanny_hash | string \| null | No |  |
| run_path | string \| null | No |  |
| py_dirs | string[] \| null | No |  |
| rint_commitment | string \| null | No |  |
| rint_nonce | string \| null | No |  |
| rint_pubkey | string \| null | No |  |
| tls_cert | string \| null | No |  |
| tls_cert_sig | string \| null | No |  |
| tls_ca_cert | string \| null | No |  |
| tls_client_cert | string \| null | No |  |
| tls_client_key | string \| null | No |  |
| tls_client_key_password | string \| null | No |  |
| e2e_pubkey | string \| null | No |  |
| cllmv_session_init | string \| null | No |  |
| env | string | Yes |  |
| code | string \| null | No |  |
| run_code | string \| null | No |  |
| inspecto | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Verify Graval Launch Config Instance

Verify Graval launch config instance by validating PoVW proof and symmetric key usage.


<div class="api-test-widget" data-widget-id="widget_put__instances_launch_config__config_id__graval"></div>
<script type="application/json" data-widget-config="widget_put__instances_launch_config__config_id__graval">{"endpoint":"/instances/launch_config/{config_id}/graval","method":"PUT","parameters":[{"name":"config_id","type":"string","required":true,"description":"","in":"path"},{"name":"Authorization","type":"string","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `PUT /instances/launch_config/{config_id}/graval`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| config_id | string | Yes |  |
| Authorization | string | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Activate Launch Config Instance


<div class="api-test-widget" data-widget-id="widget_get__instances_launch_config__config_id__activate"></div>
<script type="application/json" data-widget-config="widget_get__instances_launch_config__config_id__activate">{"endpoint":"/instances/launch_config/{config_id}/activate","method":"GET","parameters":[{"name":"config_id","type":"string","required":true,"description":"","in":"path"},{"name":"Authorization","type":"string","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /instances/launch_config/{config_id}/activate`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| config_id | string | Yes |  |
| Authorization | string | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Get Instance Nonce

Generate a nonce for TEE instance verification.

This endpoint is called by chute instances during TEE verification (Phase 1).
The nonce is used to bind the attestation evidence to this specific verification request.


<div class="api-test-widget" data-widget-id="widget_get__instances_nonce"></div>
<script type="application/json" data-widget-config="widget_get__instances_nonce">{"endpoint":"/instances/nonce","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /instances/nonce`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Token


<div class="api-test-widget" data-widget-id="widget_get__instances_token_check"></div>
<script type="application/json" data-widget-config="widget_get__instances_token_check">{"endpoint":"/instances/token_check","method":"GET","parameters":[{"name":"salt","type":"string","required":false,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /instances/token_check`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| salt | string | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Get Tee Instance Evidence

Get TEE evidence for a specific instance (TDX quote, GPU evidence, certificate).

Args:
    instance_id: Instance ID
    nonce: User-provided nonce (64 hex characters, 32 bytes)

Returns:
    TeeInstanceEvidence with quote, gpu_evidence, and certificate

Raises:
    404: Instance not found
    400: Invalid nonce format or instance not TEE-enabled
    403: User cannot access instance
    429: Rate limit exceeded
    500: Server attestation failures


<div class="api-test-widget" data-widget-id="widget_get__instances__instance_id__evidence"></div>
<script type="application/json" data-widget-config="widget_get__instances__instance_id__evidence">{"endpoint":"/instances/{instance_id}/evidence","method":"GET","requiresAuth":true,"parameters":[{"name":"instance_id","type":"string","required":true,"description":"","in":"path"},{"name":"nonce","type":"string","required":true,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /instances/{instance_id}/evidence`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| instance_id | string | Yes |  |
| nonce | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Stream Logs

Fetch raw kubernetes pod logs.

NOTE: These are pod logs, not request data/etc., so it will never
include prompts, responses, etc. Used for troubleshooting and checking
status of warmup, etc.


<div class="api-test-widget" data-widget-id="widget_get__instances__instance_id__logs"></div>
<script type="application/json" data-widget-config="widget_get__instances__instance_id__logs">{"endpoint":"/instances/{instance_id}/logs","method":"GET","requiresAuth":true,"parameters":[{"name":"instance_id","type":"string","required":true,"description":"","in":"path"},{"name":"backfill","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /instances/{instance_id}/logs`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| instance_id | string | Yes |  |
| backfill | integer \| null | No |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Disable Instance Endpoint


<div class="api-test-widget" data-widget-id="widget_post__instances__chute_id___instance_id__disable"></div>
<script type="application/json" data-widget-config="widget_post__instances__chute_id___instance_id__disable">{"endpoint":"/instances/{chute_id}/{instance_id}/disable","method":"POST","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"instance_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /instances/{chute_id}/{instance_id}/disable`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
| instance_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Delete Instance


<div class="api-test-widget" data-widget-id="widget_delete__instances__chute_id___instance_id_"></div>
<script type="application/json" data-widget-config="widget_delete__instances__chute_id___instance_id_">{"endpoint":"/instances/{chute_id}/{instance_id}","method":"DELETE","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"instance_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /instances/{chute_id}/{instance_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
| instance_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---
