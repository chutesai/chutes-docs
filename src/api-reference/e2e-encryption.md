# E2e Encryption API Reference

This section covers all endpoints related to e2e encryption.


## Get E2E Instances

Discover E2E-capable instances for a chute and get nonces for invocation.


<div class="api-test-widget" data-widget-id="widget_get__e2e_instances__chute_id_"></div>
<script type="application/json" data-widget-config="widget_get__e2e_instances__chute_id_">{"endpoint":"/e2e/instances/{chute_id}","method":"GET","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /e2e/instances/{chute_id}`

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

## E2E Invoke

Relay an E2E encrypted invocation to a specific instance.


<div class="api-test-widget" data-widget-id="widget_post__e2e_invoke"></div>
<script type="application/json" data-widget-config="widget_post__e2e_invoke">{"endpoint":"/e2e/invoke","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chute-Id","type":"string","required":true,"description":"","in":"header"},{"name":"X-Instance-Id","type":"string","required":true,"description":"","in":"header"},{"name":"X-E2E-Nonce","type":"string","required":true,"description":"","in":"header"},{"name":"X-E2E-Stream","type":"string","required":false,"description":"","in":"header"},{"name":"X-E2E-Path","type":"string","required":false,"description":"","in":"header"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /e2e/invoke`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| X-Chute-Id | string | Yes |  |
| X-Instance-Id | string | Yes |  |
| X-E2E-Nonce | string | Yes |  |
| X-E2E-Stream | string | No |  |
| X-E2E-Path | string | No |  |
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
