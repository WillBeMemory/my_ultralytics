"""Final verification: C2f_Simple ≡ C3k2(n=2, c3k=False)."""
import torch
from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.FPN_PAN_BiFPN import C2f_Simple, FPN_PAN_BiFPN

configs = [
    (768, 256, 2, "fpn_p3"),
    (512, 128, 2, "fpn_p2"),
    (512, 256, 2, "pan_p3"),
    (1024, 512, 2, "pan_p4"),
]

print("=== C2f_Simple vs C3k2(n=2, c3k=False) ===")
all_ok = True
for c1, c2, n, name in configs:
    torch.manual_seed(42)
    m_c3k2 = C3k2(c1, c2, n=n, c3k=False)
    torch.manual_seed(42)
    m_simple = C2f_Simple(c1, c2, n=n)
    p_c3k2 = sum(p.numel() for p in m_c3k2.parameters())
    p_simple = sum(p.numel() for p in m_simple.parameters())
    weight_match = all(torch.equal(m_c3k2.state_dict()[k], m_simple.state_dict()[k])
                       for k in m_c3k2.state_dict() if 'running' not in k and 'num_batches' not in k)
    x = torch.randn(1, c1, 32, 32)
    m_c3k2.eval(); m_simple.eval()
    with torch.no_grad():
        out_match = torch.allclose(m_c3k2(x), m_simple(x), atol=1e-6)
    status = "OK" if weight_match and out_match else "FAIL"
    if not weight_match or not out_match:
        all_ok = False
    print(f"  {name}: params={p_c3k2:,} / {p_simple:,}  weights={weight_match}  output={out_match}  [{status}]")

print(f"\nFinal: {'ALL OK' if all_ok else 'SOME FAILED'}")

# Full FPN_PAN_BiFPN
print("\n=== FPN_PAN_BiFPN ===")
model = FPN_PAN_BiFPN([256, 512, 512], [128, 256, 512], num_bifpn_layers=1, use_refine=False)
p2 = torch.randn(1, 256, 80, 80)
p3 = torch.randn(1, 512, 40, 40)
p4 = torch.randn(1, 512, 20, 20)
model.eval()
with torch.no_grad():
    out = model([p2, p3, p4])
for i, o in enumerate(out):
    print(f"  P{i+2}: {o.shape}")
print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")