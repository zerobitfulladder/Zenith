# Static 3-layer conv Zenith + label memory

L1 4x4/s2 K=64 -> 13x13 | L2 3x3/s2 K=64 -> 6x6 | L3 3x3/s1 K=100 -> 4x4 | TOP K=200

- probe_L1: 0.9516
- probe_L2: 0.9582
- probe_L3: 0.9578
- hard_readout: 0.8706
- memories_per_label: {'0': 15, '1': 15, '2': 24, '3': 23, '4': 24, '5': 21, '6': 21, '7': 24, '8': 17, '9': 16}
- l1_units_used: 64
- l2_units_used: 64
- l3_units_used: 100
- top_units_used: 200
- label_query_hits_own_memory: 10/10
- roundtrip_corr: 0.487
