
from typing import Optional
from web3 import Web3

ABI_VALIDATOR = [
    {"inputs":[{"internalType":"bytes32","name":"action","type":"bytes32"},{"internalType":"uint256","name":"score","type":"uint256"}],
     "name":"recordDecision","outputs":[],"stateMutability":"nonpayable","type":"function"}
]

class ChainClient:
    def __init__(self, rpc: str, chain_id: int, private_key: Optional[str], address: str):
        self.w3 = Web3(Web3.HTTPProvider(rpc))
        self.chain_id = chain_id
        self.addr = address
        self.acc = self.w3.eth.account.from_key(private_key) if private_key else None
        self.sc = self.w3.eth.contract(address=self.addr, abi=ABI_VALIDATOR) if self.addr else None

    def submit_decision(self, action: str, score: float) -> str:
        if not self.sc or not self.acc:
            return "0x0"
        tx = self.sc.functions.recordDecision(
            Web3.keccak(text=action), int(max(0.0, min(score, 1.0))*1_000_000)
        ).build_transaction({
            "chainId": self.chain_id,
            "nonce": self.w3.eth.get_transaction_count(self.acc.address),
            "gas": 200000,
            "maxFeePerGas": self.w3.to_wei("2", "gwei"),
            "maxPriorityFeePerGas": self.w3.to_wei("1", "gwei"),
        })
        signed = self.acc.sign_transaction(tx)
        txh = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        return self.w3.to_hex(txh)
