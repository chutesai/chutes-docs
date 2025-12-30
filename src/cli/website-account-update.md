# Updating an account made on the website

If you created your account from the website and now wish to use the CLI you will need to follow this guide. Through it you will create a bittensor wallet and a Chutes config file and sync that info with your account. 

## Updating an account made with the website.

There are several steps required to use the CLI if you originally registered on the website and did not provide a hotkey/coldkey.

To begin if you do not already have a bittensor wallet you need to install the Bittensor CLI and create one

## Install the Bittensor CLI and create a new wallet.

Install the Bittensor CLI:

```bash
pip install bittensor-cli 
```

Verify installation:

```bash
btcli --version
```
Create a new walle with coldkey and hotkey:

```bash
btcli wallet create --wallet.name <my_coldkey> --wallet.hotkey <my_hotkey>
```

You will then be prompted to configure the wallet by setting a password for the coldkey, and choosing the desired mnemonic length. Completing the prompts creates a complete Bittensor wallet by setting up both coldkey and hotkeys. A unique mnemonic is generated for each key and output to the terminal upon creation.

your new wallet can then be found here:

```bash
~/.bittensor/wallets
```
you can see the full contents like this:

```bash
tree ~/.bittensor/
```
It should look something like this.

```bash
tree ~/.bittensor/
/Users/docwriter/.bittensor/    # The Bittensor root directory.
└── wallets                     # The folder contains all Bittensor wallets.
    └── my_coldkey            # The name of the wallet.
        ├── coldkey             # The password-encrypted coldkey.
        ├── coldkeypub.txt      # The unencrypted version of the coldkey.
        └── hotkeys             # The folder contains all this coldkey's hotkeys.
            └── my_hotkey     # The unencrypted hotkey information.
```
You can then check the data in any of these files like this:

```bash
cd ~/.bittensor/wallets/test-coldkey
cat coldkeypub.txt | jq
{
  "accountId": "0x36e49805b105af2b5572cfc86426247df111df2f584767ca739d9fa085246c51",
  "publicKey": "0x36e49805b105af2b5572cfc86426247df111df2f584767ca739d9fa085246c51",
  "privateKey": null,
  "secretPhrase": null,
  "secretSeed": null,
  "ss58Address": "5DJgMDvzC27QTBfmgGQaNWBQd8CKP9z5A12yjbG6TZ5bxNE1"
}
```

Once the wallet is created you can now move on to the next step creating the config.ini file.

## Creating your Chutes config.ini file.

create a file called config.ini and place it in this folder, `~/.chutes` final path should be `~/.chutes/config.ini`

The contents of the config.ini file should be as follows:

```bash
[api]
base_url = https://api.chutes.ai

[auth]
username = me
user_id = uid
hotkey_seed = replaceme
hotkey_name = replaceme
hotkey_ss58address = replaceme

[payment]
address = replaceme

```
You can get your username and user_id with the get user info api endpoint:

```bash
curl -X GET "https://api.chutes.ai/users/me" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <chutes api key>" \
  ```

Add the username and user_id from the output of this command to the config.ini file in their designated spots.

The hotkey_name is the base file name of your hotkey. In this example it would be my_hotkey.

Next locate the required hotkey info from this location:

```bash
cd ~/.bittensor/wallets/my_coldkey/hotkeys
cat hotkeys/my_hotkey | jq
{
  "accountId": "0xc66695556006c79e278f487b01d44cf4bc611f195615a321bf3208f5e351621e",
  "publicKey": "0xc66695556006c79e278f487b01d44cf4bc611f195615a321bf3208f5e351621e",
  "privateKey": "0x38d3ae3b6e4b5df8415d15f44f * * * 0f975749f835fc221b * * * cbaac9f5ba6b1c90978e3858 * * * f0e0470be681c0b28fe2d64",
  "secretPhrase": "pyramid xxx wide slush xxx hub xxx crew spin xxx easily xxx",
  "secretSeed": "0x6c359cc52ff1256c9e5 * * * 5536c * * * 892e9ffe4e4066ad2a6e35561d6964e",
  "ss58Address": "5GYqp3eKu6W7KxhCNrHrVaPjsJHHLuAs5jbYWfeNzVudH8DE"
}

```
Update the missing fields in config.ini file with the info found here. in the hotkey_seed field place the value from secretSeed. 

(#remove the 0x prefix from the front of the secret seed before you add it to config.ini or it will not work)

In the hotkey_ss58address field place the value from ss58Address. 


Finally locate the coldkey ss58Address and put it in the address field in the payment section. 

```bash
cd ~/.bittensor/wallets/my_coldkey
cat coldkeypub.txt | jq
{
  "accountId": "0x36e49805b105af2b5572cfc86426247df111df2f584767ca739d9fa085246c51",
  "publicKey": "0x36e49805b105af2b5572cfc86426247df111df2f584767ca739d9fa085246c51",
  "privateKey": null,
  "secretPhrase": null,
  "secretSeed": null,
  "ss58Address": "5DJgMDvzC27QTBfmgGQaNWBQd8CKP9z5A12yjbG6TZ5bxNE1"
}
```

The config.ini file is not complete save it and close it.

## Update your Chutes account

The final step is to update your Chutes account with the newly created hotkey and coldkey:

```bash
curl -XPOST https://api.chutes.ai/users/change_bt_auth -H "Authorization: <fingerprint>" -H "Content-Type: application/json" -d '{"coldkey": "ss58 of the coldkey, from ~/.bittensor/wallets/your-coldkey/coldkeypub.txt", "hotkey": "ss58Address from the hotkey"}'
```
When the command completed check your Chutes account from the website and confirm that the hotkey and coldkey match those in your wallet. 

### Support Resources

- 📖 **Documentation**: [Complete Docs](/docs)
- 💬 **Discord**: [Community Chat](https://discord.gg/wHrXwWkCRz)
- 📨 **Support**: [Email](support@chutes.ai)
- 🐛 **Issues**: [GitHub Issues](https://github.com/chutesai/chutes/issues)

