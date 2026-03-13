# Servers API Reference

This section covers all endpoints related to servers.


## Get Nonce

Generate a nonce for boot attestation.

This endpoint is called by VMs during boot before any registration.
No authentication required as the VM doesn't exist in the system yet.


<div class="api-test-widget" data-widget-id="widget_get__servers_nonce"></div>
<script type="application/json" data-widget-config="widget_get__servers_nonce">{"endpoint":"/servers/nonce","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /servers/nonce`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Verify Boot Attestation

Verify boot attestation and return LUKS passphrase.

This endpoint verifies the TDX quote against expected boot measurements
and returns the LUKS passphrase for disk decryption if valid.


<div class="api-test-widget" data-widget-id="widget_post__servers_boot_attestation"></div>
<script type="application/json" data-widget-config="widget_post__servers_boot_attestation">{"endpoint":"/servers/boot/attestation","method":"POST","parameters":[{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"quote":{"type":"string","title":"Quote","description":"Base64 encoded TDX quote"},"miner_hotkey":{"type":"string","title":"Miner Hotkey","description":"Miner hotkey that owns this VM"},"vm_name":{"type":"string","title":"Vm Name","description":"VM name/identifier"}},"required":["quote","miner_hotkey","vm_name"]}}</script>

**Endpoint:** `POST /servers/boot/attestation`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| X-Chutes-Nonce | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| quote | string | Yes | Base64 encoded TDX quote |
| miner_hotkey | string | Yes | Miner hotkey that owns this VM |
| vm_name | string | Yes | VM name/identifier |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Get Cache Luks Passphrase

Retrieve existing LUKS passphrase for cache volume encryption.

This endpoint is called when the initramfs detects that the cache volume
is already encrypted. It retrieves the passphrase that was previously
generated for this VM configuration (miner_hotkey + vm_name).

The hotkey must be provided as a query parameter.
The boot token must be provided in the X-Boot-Token header.


<div class="api-test-widget" data-widget-id="widget_get__servers__vm_name__luks"></div>
<script type="application/json" data-widget-config="widget_get__servers__vm_name__luks">{"endpoint":"/servers/{vm_name}/luks","method":"GET","parameters":[{"name":"vm_name","type":"string","required":true,"description":"","in":"path"},{"name":"hotkey","type":"string","required":true,"description":"","in":"query"},{"name":"X-Boot-Token","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /servers/{vm_name}/luks`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| vm_name | string | Yes |  |
| hotkey | string | Yes |  |
| X-Boot-Token | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Sync Luks Passphrases

Sync LUKS passphrases: VM sends volume list; API returns keys for existing volumes,
creates keys for new volumes, rekeys volumes in rekey list, and prunes stored keys
for volumes not in the list. Boot token is consumed after successful POST.


<div class="api-test-widget" data-widget-id="widget_post__servers__vm_name__luks"></div>
<script type="application/json" data-widget-config="widget_post__servers__vm_name__luks">{"endpoint":"/servers/{vm_name}/luks","method":"POST","parameters":[{"name":"vm_name","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Boot-Token","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"volumes":{"items":{"type":"string"},"type":"array","title":"Volumes","description":"Volume names the VM is managing (defines full set)"},"rekey":{"anyOf":[{"items":{"type":"string"},"type":"array"},{"type":"null"}],"title":"Rekey","description":"Volume names that must receive new passphrases (no reuse); must be subset of volumes"}},"required":["volumes"]}}</script>

**Endpoint:** `POST /servers/{vm_name}/luks`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| vm_name | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Boot-Token | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| volumes | string[] | Yes | Volume names the VM is managing (defines full set) |
| rekey | string[] \| null | No | Volume names that must receive new passphrases (no reuse); must be subset of volumes |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Create Server

Register a new server.

This is called via CLI after the server has booted and decrypted its disk.
Links the server to any existing boot attestation history via server ip.


<div class="api-test-widget" data-widget-id="widget_post__servers_"></div>
<script type="application/json" data-widget-config="widget_post__servers_">{"endpoint":"/servers/","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"host":{"type":"string","title":"Host","description":"Public IP address or DNS Name of the server"},"id":{"type":"string","title":"Id","description":"Server ID (e.g. k8s node uid)"},"name":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Name","description":"Server name (defaults to server id if omitted)"},"gpus":{"items":{"$ref":"#/components/schemas/NodeArgs"},"type":"array","title":"Gpus","description":"GPU info for this server"}},"required":["host","id","gpus"]}}</script>

**Endpoint:** `POST /servers/`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| host | string | Yes | Public IP address or DNS Name of the server |
| id | string | Yes | Server ID (e.g. k8s node uid) |
| name | string \| null | No | Server name (defaults to server id if omitted) |
| gpus | NodeArgs[] | Yes | GPU info for this server |


### Responses

| Status Code | Description |
|-------------|-------------|
| 201 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## List User Servers

List all servers for the authenticated miner.


<div class="api-test-widget" data-widget-id="widget_get__servers_"></div>
<script type="application/json" data-widget-config="widget_get__servers_">{"endpoint":"/servers/","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /servers/`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
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

## Patch Server Name

Update name for an existing server. Path is server_id; query param is the new name.
The server row is updated when hotkey and server_id match.


<div class="api-test-widget" data-widget-id="widget_patch__servers__server_id_"></div>
<script type="application/json" data-widget-config="widget_patch__servers__server_id_">{"endpoint":"/servers/{server_id}","method":"PATCH","requiresAuth":true,"parameters":[{"name":"server_id","type":"string","required":true,"description":"","in":"path"},{"name":"server_name","type":"string","required":true,"description":"New VM name to set","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `PATCH /servers/{server_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| server_id | string | Yes |  |
| server_name | string | Yes | New VM name to set |
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

## Get Server Details

Get details for a specific server by miner hotkey and server id.


<div class="api-test-widget" data-widget-id="widget_get__servers__server_id_"></div>
<script type="application/json" data-widget-config="widget_get__servers__server_id_">{"endpoint":"/servers/{server_id}","method":"GET","requiresAuth":true,"parameters":[{"name":"server_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /servers/{server_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| server_id | string | Yes |  |
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

## Remove Server

Remove a server by miner hotkey and server id or VM name (path param server_name_or_id).


<div class="api-test-widget" data-widget-id="widget_delete__servers__server_name_or_id_"></div>
<script type="application/json" data-widget-config="widget_delete__servers__server_name_or_id_">{"endpoint":"/servers/{server_name_or_id}","method":"DELETE","requiresAuth":true,"parameters":[{"name":"server_name_or_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /servers/{server_name_or_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| server_name_or_id | string | Yes |  |
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

## Get Runtime Nonce

Generate a nonce for runtime attestation.


<div class="api-test-widget" data-widget-id="widget_get__servers__server_id__nonce"></div>
<script type="application/json" data-widget-config="widget_get__servers__server_id__nonce">{"endpoint":"/servers/{server_id}/nonce","method":"GET","requiresAuth":true,"parameters":[{"name":"server_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /servers/{server_id}/nonce`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| server_id | string | Yes |  |
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

## Verify Runtime Attestation

Verify runtime attestation with full measurement validation.


<div class="api-test-widget" data-widget-id="widget_post__servers__server_id__attestation"></div>
<script type="application/json" data-widget-config="widget_post__servers__server_id__attestation">{"endpoint":"/servers/{server_id}/attestation","method":"POST","requiresAuth":true,"parameters":[{"name":"server_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"quote":{"type":"string","title":"Quote","description":"Base64 encoded TDX quote"}},"required":["quote"]}}</script>

**Endpoint:** `POST /servers/{server_id}/attestation`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| server_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| quote | string | Yes | Base64 encoded TDX quote |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Get Attestation Status

Get current attestation status for a server by miner hotkey and server id.


<div class="api-test-widget" data-widget-id="widget_get__servers__server_id__attestation_status"></div>
<script type="application/json" data-widget-config="widget_get__servers__server_id__attestation_status">{"endpoint":"/servers/{server_id}/attestation/status","method":"GET","requiresAuth":true,"parameters":[{"name":"server_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /servers/{server_id}/attestation/status`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| server_id | string | Yes |  |
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
