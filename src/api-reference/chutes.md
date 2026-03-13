# Chutes API Reference

This section covers all endpoints related to chutes.


## Share Chute

Share a chute with another user.


<div class="api-test-widget" data-widget-id="widget_post__chutes_share"></div>
<script type="application/json" data-widget-config="widget_post__chutes_share">{"endpoint":"/chutes/share","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"chute_id_or_name":{"type":"string","title":"Chute Id Or Name"},"user_id_or_name":{"type":"string","title":"User Id Or Name"}},"required":["chute_id_or_name","user_id_or_name"]}}</script>

**Endpoint:** `POST /chutes/share`

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
| chute_id_or_name | string | Yes |  |
| user_id_or_name | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Unshare Chute

Unshare a chute with another user.


<div class="api-test-widget" data-widget-id="widget_post__chutes_unshare"></div>
<script type="application/json" data-widget-config="widget_post__chutes_unshare">{"endpoint":"/chutes/unshare","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"chute_id_or_name":{"type":"string","title":"Chute Id Or Name"},"user_id_or_name":{"type":"string","title":"User Id Or Name"}},"required":["chute_id_or_name","user_id_or_name"]}}</script>

**Endpoint:** `POST /chutes/unshare`

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
| chute_id_or_name | string | Yes |  |
| user_id_or_name | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Make Public

Promote subnet chutes to public visibility, owned by the calling subnet admin user.


<div class="api-test-widget" data-widget-id="widget_post__chutes_make_public"></div>
<script type="application/json" data-widget-config="widget_post__chutes_make_public">{"endpoint":"/chutes/make_public","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"chutes":{"items":{"type":"string"},"type":"array","title":"Chutes"}},"required":["chutes"]}}</script>

**Endpoint:** `POST /chutes/make_public`

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
| chutes | string[] | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## List Boosted Chutes

Get a list of chutes that have a boost.


<div class="api-test-widget" data-widget-id="widget_get__chutes_boosted"></div>
<script type="application/json" data-widget-config="widget_get__chutes_boosted">{"endpoint":"/chutes/boosted","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /chutes/boosted`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## List Available Affine Chutes

Get a list of affine chutes where the creator/user has a non-zero balance.


<div class="api-test-widget" data-widget-id="widget_get__chutes_affine_available"></div>
<script type="application/json" data-widget-config="widget_get__chutes_affine_available">{"endpoint":"/chutes/affine_available","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /chutes/affine_available`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## List Chutes

List (and optionally filter/paginate) chutes.


<div class="api-test-widget" data-widget-id="widget_get__chutes_"></div>
<script type="application/json" data-widget-config="widget_get__chutes_">{"endpoint":"/chutes/","method":"GET","requiresAuth":true,"parameters":[{"name":"include_public","type":"boolean \\| null","required":false,"description":"","in":"query"},{"name":"template","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"name","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"exclude","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"image","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"slug","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"page","type":"integer","required":false,"description":"","in":"query"},{"name":"limit","type":"integer","required":false,"description":"","in":"query"},{"name":"offset","type":"integer","required":false,"description":"","in":"query"},{"name":"include_schemas","type":"boolean \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /chutes/`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| include_public | boolean \| null | No |  |
| template | string \| null | No |  |
| name | string \| null | No |  |
| exclude | string \| null | No |  |
| image | string \| null | No |  |
| slug | string \| null | No |  |
| page | integer | No |  |
| limit | integer | No |  |
| offset | integer | No |  |
| include_schemas | boolean \| null | No |  |
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

## Deploy Chute

Standard deploy from the CDK.


<div class="api-test-widget" data-widget-id="widget_post__chutes_"></div>
<script type="application/json" data-widget-config="widget_post__chutes_">{"endpoint":"/chutes/","method":"POST","requiresAuth":true,"parameters":[{"name":"accept_fee","type":"boolean \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"name":{"type":"string","maxLength":128,"minLength":3,"title":"Name"},"tagline":{"anyOf":[{"type":"string","maxLength":1024},{"type":"null"}],"title":"Tagline","default":""},"readme":{"anyOf":[{"type":"string","maxLength":16384},{"type":"null"}],"title":"Readme","default":""},"tool_description":{"anyOf":[{"type":"string","maxLength":16384},{"type":"null"}],"title":"Tool Description"},"logo_id":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Logo Id"},"image":{"type":"string","title":"Image"},"public":{"type":"boolean","title":"Public"},"code":{"type":"string","maxLength":150000,"title":"Code"},"filename":{"type":"string","title":"Filename"},"ref_str":{"type":"string","title":"Ref Str"},"standard_template":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Standard Template"},"node_selector":{"$ref":"#/components/schemas/NodeSelector"},"cords":{"anyOf":[{"items":{"$ref":"#/components/schemas/Cord"},"type":"array"},{"type":"null"}],"title":"Cords","default":[]},"jobs":{"anyOf":[{"items":{"$ref":"#/components/schemas/Job"},"type":"array"},{"type":"null"}],"title":"Jobs","default":[]},"concurrency":{"anyOf":[{"type":"integer","maximum":256},{"type":"null"}],"title":"Concurrency","gte":0},"revision":{"anyOf":[{"type":"string","pattern":"^[a-fA-F0-9]{40}$"},{"type":"null"}],"title":"Revision"},"max_instances":{"anyOf":[{"type":"integer","maximum":100,"minimum":1},{"type":"null"}],"title":"Max Instances","default":1},"scaling_threshold":{"anyOf":[{"type":"number","maximum":1,"minimum":0},{"type":"null"}],"title":"Scaling Threshold","default":0.75},"shutdown_after_seconds":{"anyOf":[{"type":"integer","maximum":604800,"minimum":60},{"type":"null"}],"title":"Shutdown After Seconds","default":300},"allow_external_egress":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Allow External Egress","default":false},"encrypted_fs":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Encrypted Fs","default":false},"tee":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Tee","default":false},"lock_modules":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Lock Modules"}},"required":["name","image","public","code","filename","ref_str","node_selector"]}}</script>

**Endpoint:** `POST /chutes/`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| accept_fee | boolean \| null | No |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | Yes |  |
| tagline | string \| null | No |  |
| readme | string \| null | No |  |
| tool_description | string \| null | No |  |
| logo_id | string \| null | No |  |
| image | string | Yes |  |
| public | boolean | Yes |  |
| code | string | Yes |  |
| filename | string | Yes |  |
| ref_str | string | Yes |  |
| standard_template | string \| null | No |  |
| node_selector | NodeSelector | Yes |  |
| cords | Cord[] \| null | No |  |
| jobs | Job[] \| null | No |  |
| concurrency | integer \| null | No |  |
| revision | string \| null | No |  |
| max_instances | integer \| null | No |  |
| scaling_threshold | number \| null | No |  |
| shutdown_after_seconds | integer \| null | No |  |
| allow_external_egress | boolean \| null | No |  |
| encrypted_fs | boolean \| null | No |  |
| tee | boolean \| null | No |  |
| lock_modules | boolean \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## List Rolling Updates


<div class="api-test-widget" data-widget-id="widget_get__chutes_rolling_updates"></div>
<script type="application/json" data-widget-config="widget_get__chutes_rolling_updates">{"endpoint":"/chutes/rolling_updates","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /chutes/rolling_updates`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Gpu Count History


<div class="api-test-widget" data-widget-id="widget_get__chutes_gpu_count_history"></div>
<script type="application/json" data-widget-config="widget_get__chutes_gpu_count_history">{"endpoint":"/chutes/gpu_count_history","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /chutes/gpu_count_history`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Chute Miner Mean Index


<div class="api-test-widget" data-widget-id="widget_get__chutes_miner_means"></div>
<script type="application/json" data-widget-config="widget_get__chutes_miner_means">{"endpoint":"/chutes/miner_means","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /chutes/miner_means`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Chute Miner Means

Load a chute's mean TPS and output token count by miner ID.


<div class="api-test-widget" data-widget-id="widget_get__chutes_miner_means__chute_id___ext_"></div>
<script type="application/json" data-widget-config="widget_get__chutes_miner_means__chute_id___ext_">{"endpoint":"/chutes/miner_means/{chute_id}.{ext}","method":"GET","parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"ext","type":"string \\| null","required":true,"description":"","in":"path"}],"requestBody":null}</script>

**Endpoint:** `GET /chutes/miner_means/{chute_id}.{ext}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
| ext | string \| null | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Get Chute Miner Means

Load a chute's mean TPS and output token count by miner ID.


<div class="api-test-widget" data-widget-id="widget_get__chutes_miner_means__chute_id_"></div>
<script type="application/json" data-widget-config="widget_get__chutes_miner_means__chute_id_">{"endpoint":"/chutes/miner_means/{chute_id}","method":"GET","parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"ext","type":"string \\| null","required":false,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /chutes/miner_means/{chute_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
| ext | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Get Chute Code

Load a chute's code by ID or name.


<div class="api-test-widget" data-widget-id="widget_get__chutes_code__chute_id_"></div>
<script type="application/json" data-widget-config="widget_get__chutes_code__chute_id_">{"endpoint":"/chutes/code/{chute_id}","method":"GET","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /chutes/code/{chute_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
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

## Get Chute Hf Info

Return Hugging Face repo_id and revision for a chute so miners can predownload the model.
Miner-only; responses are cached by chute_id via aiocache.


<div class="api-test-widget" data-widget-id="widget_get__chutes__chute_id__hf_info"></div>
<script type="application/json" data-widget-config="widget_get__chutes__chute_id__hf_info">{"endpoint":"/chutes/{chute_id}/hf_info","method":"GET","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /chutes/{chute_id}/hf_info`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
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

## Warm Up Chute

Warm up a chute.


<div class="api-test-widget" data-widget-id="widget_get__chutes_warmup__chute_id_or_name_"></div>
<script type="application/json" data-widget-config="widget_get__chutes_warmup__chute_id_or_name_">{"endpoint":"/chutes/warmup/{chute_id_or_name}","method":"GET","requiresAuth":true,"parameters":[{"name":"chute_id_or_name","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /chutes/warmup/{chute_id_or_name}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id_or_name | string | Yes |  |
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

## Get Chute Utilization

Get chute utilization data from the most recent capacity log.


<div class="api-test-widget" data-widget-id="widget_get__chutes_utilization"></div>
<script type="application/json" data-widget-config="widget_get__chutes_utilization">{"endpoint":"/chutes/utilization","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /chutes/utilization`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Tee Chute Evidence

Get TEE evidence for all instances of a chute (TDX quote, GPU evidence, certificate per instance).

Args:
    chute_id_or_name: Chute ID or name
    nonce: User-provided nonce (64 hex characters, 32 bytes)

Returns:
    TeeChuteEvidence with array of TEE instance evidence per instance

Raises:
    404: Chute not found
    400: Invalid nonce format or chute not TEE-enabled
    403: User cannot access chute
    429: Rate limit exceeded
    500: Server attestation failures


<div class="api-test-widget" data-widget-id="widget_get__chutes__chute_id_or_name__evidence"></div>
<script type="application/json" data-widget-config="widget_get__chutes__chute_id_or_name__evidence">{"endpoint":"/chutes/{chute_id_or_name}/evidence","method":"GET","requiresAuth":true,"parameters":[{"name":"chute_id_or_name","type":"string","required":true,"description":"","in":"path"},{"name":"nonce","type":"string","required":true,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /chutes/{chute_id_or_name}/evidence`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id_or_name | string | Yes |  |
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

## Get Chute

Load a chute by ID or name.


<div class="api-test-widget" data-widget-id="widget_get__chutes__chute_id_or_name_"></div>
<script type="application/json" data-widget-config="widget_get__chutes__chute_id_or_name_">{"endpoint":"/chutes/{chute_id_or_name}","method":"GET","requiresAuth":true,"parameters":[{"name":"chute_id_or_name","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /chutes/{chute_id_or_name}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id_or_name | string | Yes |  |
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

## Update Common Attributes

Update readme, tagline, etc. (but not code, image, etc.).


<div class="api-test-widget" data-widget-id="widget_put__chutes__chute_id_or_name_"></div>
<script type="application/json" data-widget-config="widget_put__chutes__chute_id_or_name_">{"endpoint":"/chutes/{chute_id_or_name}","method":"PUT","requiresAuth":true,"parameters":[{"name":"chute_id_or_name","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"tagline":{"anyOf":[{"type":"string","maxLength":1024},{"type":"null"}],"title":"Tagline","default":""},"readme":{"anyOf":[{"type":"string","maxLength":16384},{"type":"null"}],"title":"Readme","default":""},"tool_description":{"anyOf":[{"type":"string","maxLength":16384},{"type":"null"}],"title":"Tool Description","default":""},"logo_id":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Logo Id"},"max_instances":{"anyOf":[{"type":"integer","maximum":100,"minimum":1},{"type":"null"}],"title":"Max Instances","default":1},"scaling_threshold":{"anyOf":[{"type":"number","maximum":1,"minimum":0},{"type":"null"}],"title":"Scaling Threshold","default":0.75},"shutdown_after_seconds":{"anyOf":[{"type":"integer","maximum":604800,"minimum":60},{"type":"null"}],"title":"Shutdown After Seconds","default":300},"disabled":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Disabled"}},"required":[]}}</script>

**Endpoint:** `PUT /chutes/{chute_id_or_name}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id_or_name | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| tagline | string \| null | No |  |
| readme | string \| null | No |  |
| tool_description | string \| null | No |  |
| logo_id | string \| null | No |  |
| max_instances | integer \| null | No |  |
| scaling_threshold | number \| null | No |  |
| shutdown_after_seconds | integer \| null | No |  |
| disabled | boolean \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Delete Chute

Delete a chute by ID.


<div class="api-test-widget" data-widget-id="widget_delete__chutes__chute_id_"></div>
<script type="application/json" data-widget-config="widget_delete__chutes__chute_id_">{"endpoint":"/chutes/{chute_id}","method":"DELETE","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /chutes/{chute_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
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

## Easy Deploy Vllm Chute

Easy/templated vLLM deployment.


<div class="api-test-widget" data-widget-id="widget_post__chutes_vllm"></div>
<script type="application/json" data-widget-config="widget_post__chutes_vllm">{"endpoint":"/chutes/vllm","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"model":{"type":"string","title":"Model"},"logo_id":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Logo Id"},"tagline":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tagline","default":""},"tool_description":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tool Description"},"readme":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Readme","default":""},"public":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Public","default":true},"node_selector":{"anyOf":[{"$ref":"#/components/schemas/NodeSelector"},{"type":"null"}]},"engine_args":{"anyOf":[{"$ref":"#/components/schemas/VLLMEngineArgs"},{"type":"null"}]},"revision":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Revision"},"concurrency":{"anyOf":[{"type":"integer"},{"type":"null"}],"title":"Concurrency","default":8}},"required":["model"]}}</script>

**Endpoint:** `POST /chutes/vllm`

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
| model | string | Yes |  |
| logo_id | string \| null | No |  |
| tagline | string \| null | No |  |
| tool_description | string \| null | No |  |
| readme | string \| null | No |  |
| public | boolean \| null | No |  |
| node_selector | NodeSelector \| null | No |  |
| engine_args | VLLMEngineArgs \| null | No |  |
| revision | string \| null | No |  |
| concurrency | integer \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Easy Deploy Diffusion Chute

Easy/templated diffusion deployment.


<div class="api-test-widget" data-widget-id="widget_post__chutes_diffusion"></div>
<script type="application/json" data-widget-config="widget_post__chutes_diffusion">{"endpoint":"/chutes/diffusion","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"model":{"type":"string","title":"Model"},"name":{"type":"string","title":"Name"},"logo_id":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Logo Id"},"tagline":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tagline","default":""},"tool_description":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tool Description"},"readme":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Readme","default":""},"public":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Public","default":true},"node_selector":{"anyOf":[{"$ref":"#/components/schemas/NodeSelector"},{"type":"null"}]},"concurrency":{"anyOf":[{"type":"integer"},{"type":"null"}],"title":"Concurrency","default":1}},"required":["model","name"]}}</script>

**Endpoint:** `POST /chutes/diffusion`

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
| model | string | Yes |  |
| name | string | Yes |  |
| logo_id | string \| null | No |  |
| tagline | string \| null | No |  |
| tool_description | string \| null | No |  |
| readme | string \| null | No |  |
| public | boolean \| null | No |  |
| node_selector | NodeSelector \| null | No |  |
| concurrency | integer \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Teeify Chute

Create a new TEE-enabled chute from an existing affine chute.


<div class="api-test-widget" data-widget-id="widget_put__chutes__chute_id__teeify"></div>
<script type="application/json" data-widget-config="widget_put__chutes__chute_id__teeify">{"endpoint":"/chutes/{chute_id}/teeify","method":"PUT","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `PUT /chutes/{chute_id}/teeify`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
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

## Get Bounty List

List available bounties, if any.


<div class="api-test-widget" data-widget-id="widget_get__bounties_"></div>
<script type="application/json" data-widget-config="widget_get__bounties_">{"endpoint":"/bounties/","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /bounties/`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Increase Chute Bounty

Increase bounty value (creating if not exists).


<div class="api-test-widget" data-widget-id="widget_get__bounties__chute_id__increase"></div>
<script type="application/json" data-widget-config="widget_get__bounties__chute_id__increase">{"endpoint":"/bounties/{chute_id}/increase","method":"GET","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"boost","type":"number \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /bounties/{chute_id}/increase`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
| boost | number \| null | No |  |
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
