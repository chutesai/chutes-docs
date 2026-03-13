# Users API Reference

This section covers all endpoints related to users.


## Get User Growth


<div class="api-test-widget" data-widget-id="widget_get__users_growth"></div>
<script type="application/json" data-widget-config="widget_get__users_growth">{"endpoint":"/users/growth","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /users/growth`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## List Chute Shares


<div class="api-test-widget" data-widget-id="widget_get__users__user_id__shares"></div>
<script type="application/json" data-widget-config="widget_get__users__user_id__shares">{"endpoint":"/users/{user_id}/shares","method":"GET","requiresAuth":true,"parameters":[{"name":"user_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/{user_id}/shares`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| user_id | string | Yes |  |
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

## Admin User Id Lookup


<div class="api-test-widget" data-widget-id="widget_get__users_user_id_lookup"></div>
<script type="application/json" data-widget-config="widget_get__users_user_id_lookup">{"endpoint":"/users/user_id_lookup","method":"GET","requiresAuth":true,"parameters":[{"name":"username","type":"string","required":true,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/user_id_lookup`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| username | string | Yes |  |
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

## Admin Balance Lookup


<div class="api-test-widget" data-widget-id="widget_get__users__user_id_or_username__balance"></div>
<script type="application/json" data-widget-config="widget_get__users__user_id_or_username__balance">{"endpoint":"/users/{user_id_or_username}/balance","method":"GET","requiresAuth":true,"parameters":[{"name":"user_id_or_username","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/{user_id_or_username}/balance`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| user_id_or_username | string | Yes |  |
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

## Admin Invoiced User List


<div class="api-test-widget" data-widget-id="widget_get__users_invoiced_user_list"></div>
<script type="application/json" data-widget-config="widget_get__users_invoiced_user_list">{"endpoint":"/users/invoiced_user_list","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/invoiced_user_list`

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

## Admin Batch User Lookup


<div class="api-test-widget" data-widget-id="widget_post__users_batch_user_lookup"></div>
<script type="application/json" data-widget-config="widget_post__users_batch_user_lookup">{"endpoint":"/users/batch_user_lookup","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /users/batch_user_lookup`

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

## Admin Balance Change


<div class="api-test-widget" data-widget-id="widget_post__users_admin_balance_change"></div>
<script type="application/json" data-widget-config="widget_post__users_admin_balance_change">{"endpoint":"/users/admin_balance_change","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"user_id":{"type":"string","title":"User Id"},"amount":{"type":"number","title":"Amount"},"reason":{"type":"string","title":"Reason"}},"required":["user_id","amount","reason"]}}</script>

**Endpoint:** `POST /users/admin_balance_change`

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
| user_id | string | Yes |  |
| amount | number | Yes |  |
| reason | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Balance Transfer

Transfer balance from the authenticated user to a target user.
Supports three authentication methods:
  1. Hotkey authentication (X-Chutes-Hotkey + X-Chutes-Signature + X-Chutes-Nonce)
  2. Admin API key (Authorization: cpk_...)
  3. Fingerprint (Authorization: <fingerprint>)


<div class="api-test-widget" data-widget-id="widget_post__users_balance_transfer"></div>
<script type="application/json" data-widget-config="widget_post__users_balance_transfer">{"endpoint":"/users/balance_transfer","method":"POST","parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"user_id":{"type":"string","title":"User Id"},"amount":{"anyOf":[{"type":"number"},{"type":"null"}],"title":"Amount"}},"required":["user_id"]}}</script>

**Endpoint:** `POST /users/balance_transfer`

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
| user_id | string | Yes |  |
| amount | number \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Grant Subnet Role


<div class="api-test-widget" data-widget-id="widget_post__users_grant_subnet_role"></div>
<script type="application/json" data-widget-config="widget_post__users_grant_subnet_role">{"endpoint":"/users/grant_subnet_role","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"user":{"type":"string","title":"User"},"netuid":{"type":"integer","title":"Netuid"},"admin":{"type":"boolean","title":"Admin"}},"required":["user","netuid","admin"]}}</script>

**Endpoint:** `POST /users/grant_subnet_role`

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
| user | string | Yes |  |
| netuid | integer | Yes |  |
| admin | boolean | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Revoke Subnet Role


<div class="api-test-widget" data-widget-id="widget_post__users_revoke_subnet_role"></div>
<script type="application/json" data-widget-config="widget_post__users_revoke_subnet_role">{"endpoint":"/users/revoke_subnet_role","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"user":{"type":"string","title":"User"},"netuid":{"type":"integer","title":"Netuid"}},"required":["user","netuid"]}}</script>

**Endpoint:** `POST /users/revoke_subnet_role`

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
| user | string | Yes |  |
| netuid | integer | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Admin Quotas Change


<div class="api-test-widget" data-widget-id="widget_post__users__user_id__quotas"></div>
<script type="application/json" data-widget-config="widget_post__users__user_id__quotas">{"endpoint":"/users/{user_id}/quotas","method":"POST","requiresAuth":true,"parameters":[{"name":"user_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /users/{user_id}/quotas`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| user_id | string | Yes |  |
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

## Admin Get User Quotas

Load quotas for a user.


<div class="api-test-widget" data-widget-id="widget_get__users__user_id__quotas"></div>
<script type="application/json" data-widget-config="widget_get__users__user_id__quotas">{"endpoint":"/users/{user_id}/quotas","method":"GET","requiresAuth":true,"parameters":[{"name":"user_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/{user_id}/quotas`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| user_id | string | Yes |  |
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

## Admin Quota Effective Date Change


<div class="api-test-widget" data-widget-id="widget_put__users__user_id__quotas__chute_id__effective_date"></div>
<script type="application/json" data-widget-config="widget_put__users__user_id__quotas__chute_id__effective_date">{"endpoint":"/users/{user_id}/quotas/{chute_id}/effective_date","method":"PUT","requiresAuth":true,"parameters":[{"name":"user_id","type":"string","required":true,"description":"","in":"path"},{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"effective_date":{"anyOf":[{"type":"string","format":"date-time"},{"type":"null"}],"title":"Effective Date"}},"required":[]}}</script>

**Endpoint:** `PUT /users/{user_id}/quotas/{chute_id}/effective_date`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| user_id | string | Yes |  |
| chute_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| effective_date | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Admin Discounts Change


<div class="api-test-widget" data-widget-id="widget_post__users__user_id__discounts"></div>
<script type="application/json" data-widget-config="widget_post__users__user_id__discounts">{"endpoint":"/users/{user_id}/discounts","method":"POST","requiresAuth":true,"parameters":[{"name":"user_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /users/{user_id}/discounts`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| user_id | string | Yes |  |
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

## Admin List Discounts


<div class="api-test-widget" data-widget-id="widget_get__users__user_id__discounts"></div>
<script type="application/json" data-widget-config="widget_get__users__user_id__discounts">{"endpoint":"/users/{user_id}/discounts","method":"GET","requiresAuth":true,"parameters":[{"name":"user_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/{user_id}/discounts`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| user_id | string | Yes |  |
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

## Admin Enable Invoicing


<div class="api-test-widget" data-widget-id="widget_post__users__user_id__enable_invoicing"></div>
<script type="application/json" data-widget-config="widget_post__users__user_id__enable_invoicing">{"endpoint":"/users/{user_id}/enable_invoicing","method":"POST","requiresAuth":true,"parameters":[{"name":"user_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /users/{user_id}/enable_invoicing`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| user_id | string | Yes |  |
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

## My Quotas

Load quotas for the current user.


<div class="api-test-widget" data-widget-id="widget_get__users_me_quotas"></div>
<script type="application/json" data-widget-config="widget_get__users_me_quotas">{"endpoint":"/users/me/quotas","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/me/quotas`

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

## My Discounts

Load discounts for the current user.


<div class="api-test-widget" data-widget-id="widget_get__users_me_discounts"></div>
<script type="application/json" data-widget-config="widget_get__users_me_discounts">{"endpoint":"/users/me/discounts","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/me/discounts`

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

## My Price Overrides

Load price overrides for the current user.


<div class="api-test-widget" data-widget-id="widget_get__users_me_price_overrides"></div>
<script type="application/json" data-widget-config="widget_get__users_me_price_overrides">{"endpoint":"/users/me/price_overrides","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/me/price_overrides`

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

## Chute Quota Usage

Check the current quota usage for a chute.


<div class="api-test-widget" data-widget-id="widget_get__users_me_quota_usage__chute_id_"></div>
<script type="application/json" data-widget-config="widget_get__users_me_quota_usage__chute_id_">{"endpoint":"/users/me/quota_usage/{chute_id}","method":"GET","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/me/quota_usage/{chute_id}`

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

## My Subscription Usage

Get current subscription usage and caps for the authenticated user.
Returns monthly and 4-hour window usage vs limits.


<div class="api-test-widget" data-widget-id="widget_get__users_me_subscription_usage"></div>
<script type="application/json" data-widget-config="widget_get__users_me_subscription_usage">{"endpoint":"/users/me/subscription_usage","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/me/subscription_usage`

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

## Delete My User

Delete account.


<div class="api-test-widget" data-widget-id="widget_delete__users_me"></div>
<script type="application/json" data-widget-config="widget_delete__users_me">{"endpoint":"/users/me","method":"DELETE","parameters":[{"name":"Authorization","type":"string","required":true,"description":"Authorization header","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /users/me`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| Authorization | string | Yes | Authorization header |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Set Logo

Get a detailed response for the current user.


<div class="api-test-widget" data-widget-id="widget_get__users_set_logo"></div>
<script type="application/json" data-widget-config="widget_get__users_set_logo">{"endpoint":"/users/set_logo","method":"GET","requiresAuth":true,"parameters":[{"name":"logo_id","type":"string","required":true,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/set_logo`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| logo_id | string | Yes |  |
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

## Check Username

Check if a username is valid and available.


<div class="api-test-widget" data-widget-id="widget_get__users_name_check"></div>
<script type="application/json" data-widget-config="widget_get__users_name_check">{"endpoint":"/users/name_check","method":"GET","parameters":[{"name":"username","type":"string","required":true,"description":"","in":"query"},{"name":"readonly","type":"boolean \\| null","required":false,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /users/name_check`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| username | string | Yes |  |
| readonly | boolean \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Register

Register a user.


<div class="api-test-widget" data-widget-id="widget_post__users_register"></div>
<script type="application/json" data-widget-config="widget_post__users_register">{"endpoint":"/users/register","method":"POST","requiresAuth":true,"parameters":[{"name":"token","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string","required":true,"description":"The hotkey of the user","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"username":{"type":"string","title":"Username"},"coldkey":{"type":"string","title":"Coldkey"},"logo_id":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Logo Id"}},"required":["username","coldkey"]}}</script>

**Endpoint:** `POST /users/register`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| token | string \| null | No |  |
| X-Chutes-Hotkey | string | Yes | The hotkey of the user |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| username | string | Yes |  |
| coldkey | string | Yes |  |
| logo_id | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Get Registration Token

Initial form with cloudflare + hcaptcha to generate a registration token.


<div class="api-test-widget" data-widget-id="widget_get__users_registration_token"></div>
<script type="application/json" data-widget-config="widget_get__users_registration_token">{"endpoint":"/users/registration_token","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /users/registration_token`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Post Rtok

Verify hCaptcha and get a short-lived registration token.


<div class="api-test-widget" data-widget-id="widget_post__users_registration_token"></div>
<script type="application/json" data-widget-config="widget_post__users_registration_token">{"endpoint":"/users/registration_token","method":"POST","parameters":[],"requestBody":null}</script>

**Endpoint:** `POST /users/registration_token`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Admin Create User

Create a new user manually from an admin account, no bittensor stuff necessary.


<div class="api-test-widget" data-widget-id="widget_post__users_create_user"></div>
<script type="application/json" data-widget-config="widget_post__users_create_user">{"endpoint":"/users/create_user","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"username":{"type":"string","title":"Username"},"coldkey":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Coldkey"},"hotkey":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Hotkey"},"logo_id":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Logo Id"}},"required":["username"]}}</script>

**Endpoint:** `POST /users/create_user`

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
| username | string | Yes |  |
| coldkey | string \| null | No |  |
| hotkey | string \| null | No |  |
| logo_id | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Change Fingerprint

Reset a user's fingerprint using either the hotkey or coldkey.


<div class="api-test-widget" data-widget-id="widget_post__users_change_fingerprint"></div>
<script type="application/json" data-widget-config="widget_post__users_change_fingerprint">{"endpoint":"/users/change_fingerprint","method":"POST","parameters":[{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Coldkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string","required":false,"description":"Nonce","in":"header"},{"name":"X-Chutes-Signature","type":"string","required":false,"description":"Hotkey signature","in":"header"}],"requestBody":{"type":"object","properties":{"fingerprint":{"type":"string","title":"Fingerprint"}},"required":["fingerprint"]}}</script>

**Endpoint:** `POST /users/change_fingerprint`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| Authorization | string \| null | No |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Coldkey | string \| null | No |  |
| X-Chutes-Nonce | string | No | Nonce |
| X-Chutes-Signature | string | No | Hotkey signature |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| fingerprint | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Get Login Nonce

Get a nonce for hotkey signature login.
The nonce is a UUID4 string that must be signed by the user's hotkey.
Valid for 5 minutes.


<div class="api-test-widget" data-widget-id="widget_get__users_login_nonce"></div>
<script type="application/json" data-widget-config="widget_get__users_login_nonce">{"endpoint":"/users/login/nonce","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /users/login/nonce`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Login

Exchange credentials for a JWT.

Supports two authentication methods:
1. Fingerprint: {"fingerprint": "your-fingerprint"}
2. Hotkey signature: {"hotkey": "5...", "signature": "hex...", "nonce": "uuid"}

For hotkey auth, first call GET /users/login/nonce to get a nonce,
sign it with your hotkey (e.g., `btcli w sign --message <nonce>`),
then submit the hotkey, signature, and nonce.


<div class="api-test-widget" data-widget-id="widget_post__users_login"></div>
<script type="application/json" data-widget-config="widget_post__users_login">{"endpoint":"/users/login","method":"POST","parameters":[],"requestBody":null}</script>

**Endpoint:** `POST /users/login`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Change Bt Auth

Change the bittensor hotkey/coldkey associated with an account via fingerprint auth.


<div class="api-test-widget" data-widget-id="widget_post__users_change_bt_auth"></div>
<script type="application/json" data-widget-config="widget_post__users_change_bt_auth">{"endpoint":"/users/change_bt_auth","method":"POST","parameters":[{"name":"Authorization","type":"string","required":true,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /users/change_bt_auth`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| Authorization | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Update Squad Access

Enable squad access.


<div class="api-test-widget" data-widget-id="widget_put__users_squad_access"></div>
<script type="application/json" data-widget-config="widget_put__users_squad_access">{"endpoint":"/users/squad_access","method":"PUT","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `PUT /users/squad_access`

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

## List Usage

List usage summary data.


<div class="api-test-widget" data-widget-id="widget_get__users__user_id__usage"></div>
<script type="application/json" data-widget-config="widget_get__users__user_id__usage">{"endpoint":"/users/{user_id}/usage","method":"GET","requiresAuth":true,"parameters":[{"name":"user_id","type":"string","required":true,"description":"","in":"path"},{"name":"page","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"limit","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"per_chute","type":"boolean \\| null","required":false,"description":"","in":"query"},{"name":"chute_id","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"start_date","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"end_date","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/{user_id}/usage`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| user_id | string | Yes |  |
| page | integer \| null | No |  |
| limit | integer \| null | No |  |
| per_chute | boolean \| null | No |  |
| chute_id | string \| null | No |  |
| start_date | string \| null | No |  |
| end_date | string \| null | No |  |
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

## Get User Info

Get user info.


<div class="api-test-widget" data-widget-id="widget_get__users__user_id_"></div>
<script type="application/json" data-widget-config="widget_get__users__user_id_">{"endpoint":"/users/{user_id}","method":"GET","requiresAuth":true,"parameters":[{"name":"user_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/{user_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| user_id | string | Yes |  |
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
