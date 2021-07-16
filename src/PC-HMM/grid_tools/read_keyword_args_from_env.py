import os

if __name__ == "__main__":
    for key in sorted(os.environ.keys()):
        if key[0].islower() or (
            key.startswith("HP_")
            or key.startswith("DataP_")
            or key.startswith("InitP_")
            or key.startswith("AlgP_")
        ):
            val = os.environ[key]

            # Manually remove some unnecessary env vars
            if key.startswith("rvm") or val.count(" ") > 0:
                continue
            if key == "escape_flag":
                continue

            assert key.count(" ") == 0
            assert val.count(" ") == 0
            print("--%s %s" % (key, val))
