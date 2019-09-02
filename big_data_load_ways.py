# 提取行为数据,方法一
def get_from_action_data(fname, chunk_size=50000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[
                ["user_id", "sku_id", "type", "time"]]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")

    df_ac = pd.concat(chunks, ignore_index=True)
    # type=4,为购买
    df_ac = df_ac[df_ac['type'] == 4]

    return df_ac[["user_id", "sku_id", "time"]]
    
# 提取行为数据,方法二
def get_action_data(file_name,chunk_size):
    reader = pd.read_csv(file_name,chunksize = chunk_size,iterator = True)
    chunks = []
    for chunk in reader:
        chunks.append(chunk[["user_id", "sku_id", "type", "time","cate"]])
    act_data = pd.concat(chunks,ignore_index = True)
    # 品牌==8的用户行为
    act_data = act_data[act_data["cate"] == 8]
    # 购买（type==4）行为
    act_data = act_data[act_data["type"] == 4]
    return act_data[["user_id","sku_id","time"]]
    
# 提取行为数据,方法三
def get_data_from_cache(dump_path):
    act_cate8_data = pickle.load(open(dump_path, "wb"))
    act_data = act_cate8_data[["user_id", "sku_id", "type", "time"]]
    act_data = act_data[act_data["type"] == 4]
    return act_data
