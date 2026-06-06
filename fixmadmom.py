import os
import site

site_packages = site.getsitepackages()[0]
madmom_path = os.path.join(site_packages, "madmom")

print("Patching madmom at:", madmom_path)

for root, dirs, files in os.walk(madmom_path):
    for file in files:
        if file.endswith((".py", ".pyx")):  # 🔥 incluye pyx
            filepath = os.path.join(root, file)

            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            new_content = content

            # collections fix
            new_content = new_content.replace(
                "from collections import MutableSequence",
                "from collections.abc import MutableSequence"
            )

            # numpy fixes (más completos)
            replacements = {
                "np.float": "float",
                "np.int": "int",
                "np.bool": "bool",
                "numpy.float": "float",
                "numpy.int": "int",
                "numpy.bool": "bool",
            }

            for old, new in replacements.items():
                new_content = new_content.replace(old, new)

            if new_content != content:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(new_content)

print("madmom FULL patched ✅")