import pandas as pd

METADATA_PATH = 'SraRunTable.csv'
RUN_LIST_SAVE_PATH = 'SRR_list.txt'
RUN_TO_POOL_SAVE_PATH = 'SRR_pools_map.txt'

meta_df = pd.read_csv(METADATA_PATH)
runs = meta_df['Run'].tolist()
print(meta_df.head())

# First make a basic list of all the runs (1 per line).
# These will be used for the initial download.
with open(RUN_LIST_SAVE_PATH, 'w') as f:
    lines = [run + '\n' if i <= len(runs) - 1 else run for i, run in enumerate(runs)]
    f.writelines(lines)

mouse_ids = meta_df['mouse_id'].tolist()

# Convert each set of mouse ids into a pool name
def mouse_ids_to_str(ids):
    all_ids = ids.split(',')
    all_ids = [str(int(i)) for i in all_ids]
    return '-'.join(all_ids)

mouse_pools = [mouse_ids_to_str(ids) for ids in mouse_ids]

# Assign an incrementing lane to each pool sample based on its pool (so cellranger concats all samples in a pool)
def assign_lanes(pools):
    lanes = []
    pools_seen_count = {}

    for pool in pools:
        # Increment how many times we've seen the pool
        if pool not in pools_seen_count:
            pools_seen_count[pool] = 1
        else:
            pools_seen_count[pool] += 1

        # Create a lane tag based on number of times seen as L00n
        count = str(pools_seen_count[pool])
        lane_name = 'L' + '0'*(3 - len(count)) + count
        lanes.append(lane_name)

    return lanes

lanes = assign_lanes(mouse_pools)

# Now that we have pool names for each SRR number, we can create a tsv mapping SRR name to pool name
srr_to_pool_df = pd.DataFrame(index=runs, data={'pools': mouse_pools, 'lanes': lanes})
srr_to_pool_df.to_csv(RUN_TO_POOL_SAVE_PATH, sep='\t', header=False)
print(srr_to_pool_df.head())