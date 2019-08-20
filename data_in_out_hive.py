'''写日志'''

import datetime
today = datetime.date.today()
date = today - datetime.timedelta(days=20)
date = str(date)
test_file = open('d:/test.txt','w')
test_file.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'\n')

test_file.close()


'''读取hive数据'''

##self_plan


class result_in_out_hive():

    def result_out_hive(self):
        hive_commond = ('''hive -e "select dt,gameid,sum(pay_total) pay_total from gamelive.gamelive_user_consume_detail
                    where dt >= '2018-01-01'
                    and from_source in ('present','new_noble','new_guard')
                    and gameid in (1,2336,1663,2165,2168,2135,3203,2793,1964,4,2,393,1732,6,2633)
                    and pay_desc not in ('续费')
                    group by gameid,dt"''')
        output = os.popen(hive_commond)
        result = pd.read_csv(StringIO(output.read()), sep="\t",header=0,names = ['dt','gameid','pay_total'])
        result.to_csv('/data/datacenter/tongxin_hive/timeseries_mul/data/result_import.csv',index = 0)

    def result_in_hive(self):
        hive_commond = ('''hive -e "LOAD DATA LOCAL INPATH '/data/datacenter/tongxin_hive/timeseries_mul/result/result.csv' OVERWRITE  INTO TABLE zhgametemp.time_series_result_mul;ALTER TABLE zhgametemp.time_series_result_mul SET SERDEP
ROPERTIES ('field.delim' = ',')"''')
        os.system(hive_commond)
		
		
##other_plan


import os

columns = '*'
table = 'gamelive.to_main_games_day'
date = '2019-08-18'
data_file2 = '/data/datacenter/tongxin_hive/timeseries/test.csv'
sql1 = "select %s from %s where dt='%s' limit 10" %(columns, table, date)
hive_command2 = ('''hive -e "%s" >> %s''' % (sql1, data_file2))
os.system(hive_command2)

