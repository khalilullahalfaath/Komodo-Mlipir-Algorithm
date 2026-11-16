"""Copy komodo_mlipir-1.0.1/*.m into comparison/m_orig and comparison/m_fix and patch them.

m_orig: original code, only the Foxholes label in Evaluation.m fixed (case 16 -> 14).
m_fix : m_orig + the AllHQ index fix in KMA2D.m (second stage).
"""
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent
src = root.parent / "komodo_mlipir-1.0.1"

for name in ("m_orig", "m_fix"):
    d = root / name
    d.mkdir(exist_ok=True)
    for f in src.glob("*.m"):
        shutil.copy(f, d / f.name)
    for f in ("run_kma.m", "eval_cmp.m"):
        shutil.copy(root / f, d / f)
    ev = d / "Evaluation.m"
    s = ev.read_text()
    assert "%Foxholes\ncase 16" in s
    ev.write_text(s.replace("%Foxholes\ncase 16", "%Foxholes\ncase 14", 1))

kma = root / "m_fix" / "KMA2D.m"
s = kma.read_text()
old = """            AllHQ(ind:ind+NumBM-1,:) = BigMales;
            AllHQFX(ind:ind+NumBM-1) = BigMalesFX;"""
new = """            qq = (ind-1)/SwarmSize*NumBM + 1;
            AllHQ(qq:qq+NumBM-1,:) = BigMales;
            AllHQFX(qq:qq+NumBM-1) = BigMalesFX;"""
assert old in s
kma.write_text(s.replace(old, new))
