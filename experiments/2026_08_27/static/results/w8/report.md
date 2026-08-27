# Static 3-layer conv Zenith + label memory

L1 8x8/s2 K=64 -> 11x11 | L2 3x3/s2 K=64 -> 5x5 | L3 3x3/s1 K=100 -> 3x3 | TOP K=200

- probe_L1: 0.9594
- probe_L2: 0.9606
- probe_L3: 0.9548
- hard_readout: 0.8762
- memories_per_label: {'0': 15, '1': 15, '2': 24, '3': 23, '4': 24, '5': 21, '6': 21, '7': 24, '8': 17, '9': 16}
- l1_units_used: 64
- l2_units_used: 64
- l3_units_used: 100
- top_units_used: 200
- label_query_hits_own_memory: 10/10
- roundtrip_corr: 0.651
