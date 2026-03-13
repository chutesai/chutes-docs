# Secret API Reference

This section covers all endpoints related to secret.


## List Secrets

List secrets.


<div class="api-test-widget" data-widget-id="widget_get__secrets_"></div>
<script type="application/json" data-widget-config="widget_get__secrets_">{"endpoint":"/secrets/","method":"GET","requiresAuth":true,"parameters":[{"name":"page","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"limit","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /secrets/`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| page | integer \| null | No |  |
| limit | integer \| null | No |  |
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

## Create Secret

Create a secret (e.g. private HF token).


<div class="api-test-widget" data-widget-id="widget_post__secrets_"></div>
<script type="application/json" data-widget-config="widget_post__secrets_">{"endpoint":"/secrets/","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"purpose":{"type":"string","title":"Purpose"},"key":{"type":"string","title":"Key"},"value":{"type":"string","title":"Value"}},"required":["purpose","key","value"]}}</script>

**Endpoint:** `POST /secrets/`

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
| purpose | string | Yes |  |
| key | string | Yes |  |
| value | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Get Secret

Load a single secret by ID.


<div class="api-test-widget" data-widget-id="widget_get__secrets__secret_id_"></div>
<script type="application/json" data-widget-config="widget_get__secrets__secret_id_">{"endpoint":"/secrets/{secret_id}","method":"GET","requiresAuth":true,"parameters":[{"name":"secret_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /secrets/{secret_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| secret_id | string | Yes |  |
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

## Delete Secret

Delete a secret by ID.


<div class="api-test-widget" data-widget-id="widget_delete__secrets__secret_id_"></div>
<script type="application/json" data-widget-config="widget_delete__secrets__secret_id_">{"endpoint":"/secrets/{secret_id}","method":"DELETE","requiresAuth":true,"parameters":[{"name":"secret_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /secrets/{secret_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| secret_id | string | Yes |  |
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
