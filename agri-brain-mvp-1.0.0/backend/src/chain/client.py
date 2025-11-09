from web3 import Web3
class ChainClient:
    def __init__(self, rpc=None, addresses=None, chain_id=31337, private_key=None):
        self.rpc=rpc; self.addr=addresses or {}; self.chain_id=chain_id; self.pk=private_key
        self.w3=Web3(Web3.HTTPProvider(rpc)) if rpc else None
    def available(self):
        try: return self.w3 is not None and self.w3.is_connected()
        except: return False
    def log_decision(self, agent, action, slca, metaURI=''):
        return '0xdecaf' if self.available() else '0x0'
