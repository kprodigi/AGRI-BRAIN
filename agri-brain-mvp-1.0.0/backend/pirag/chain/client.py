
import os
from typing import Optional
try:
    from web3 import Web3
except Exception:
    Web3 = None
ABI = []  # Fill post-compile; minimal placeholder
CONTRACT_ADDRESS = os.getenv("PROVENANCE_ADDR", "")
def anchor_root(root_hex: str, policy_uri: str = "") -> Optional[str]:
    if Web3 is None or not CONTRACT_ADDRESS:
        return None
    w3 = Web3(Web3.HTTPProvider(os.getenv("CHAIN_RPC","http://localhost:8545")))
    acct = w3.eth.account.from_key(os.getenv("CHAIN_PRIVKEY","0x"+"0"*64))
    contract = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=ABI)
    tx = contract.functions.anchor(bytes.fromhex(root_hex), policy_uri).build_transaction({
        "from": acct.address,
        "nonce": w3.eth.get_transaction_count(acct.address),
        "gas": 500000,
        "maxFeePerGas": w3.to_wei("2", "gwei"),
        "maxPriorityFeePerGas": w3.to_wei("1", "gwei"),
        "chainId": w3.eth.chain_id,
    })
    signed = acct.sign_transaction(tx)
    txh = w3.eth.send_raw_transaction(signed.rawTransaction)
    return txh.hex()
