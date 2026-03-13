# Model Aliases API Reference

This section covers all endpoints related to model aliases.


## List Aliases


<div class="api-test-widget" data-widget-id="widget_get__model_aliases_"></div>
<script type="application/json" data-widget-config="widget_get__model_aliases_">{"endpoint":"/model_aliases/","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /model_aliases/`

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

## Create Or Update Alias


<div class="api-test-widget" data-widget-id="widget_post__model_aliases_"></div>
<script type="application/json" data-widget-config="widget_post__model_aliases_">{"endpoint":"/model_aliases/","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"alias":{"type":"string","title":"Alias"},"chute_ids":{"items":{"type":"string"},"type":"array","title":"Chute Ids"}},"required":["alias","chute_ids"]}}</script>

**Endpoint:** `POST /model_aliases/`

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
| alias | string | Yes |  |
| chute_ids | string[] | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 201 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Delete Alias


<div class="api-test-widget" data-widget-id="widget_delete__model_aliases__alias_"></div>
<script type="application/json" data-widget-config="widget_delete__model_aliases__alias_">{"endpoint":"/model_aliases/{alias}","method":"DELETE","requiresAuth":true,"parameters":[{"name":"alias","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /model_aliases/{alias}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| alias | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 204 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---
