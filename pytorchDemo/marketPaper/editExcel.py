#!/usr/bin/python3
import warnings
from datetime import datetime
import openpyxl
 
# xlsx数据源编程，参考https://blog.csdn.net/weixin_52058417/article/details/123266853


# warnings.filterwarnings("ignore")

# 加载存在的工作簿：
wb = openpyxl.load_workbook("D:/办公/DCOS/项目进度/云平台支撑域2023年项目规划.xlsx")
# 备份文件
wb.save("D:/办公/DCOS/项目进度/云平台支撑域2023年项目规划 - python-"+ datetime.now().strftime('%Y%m%d%H%M%S%f') +".xlsx")
 
# 打开指定的工作簿中的指定工作表：
ws = wb["项目规划表"]
 
ws = wb.active  # 打开激活的工作表

# 显示工作表表名：worksheets会以列表的形式返回当前工作簿里所有的工作表表名：
sheet_list = wb.worksheets
 
# 获取工作表名称：
for i in sheet_list:
    # title：获取工作表名称
    print(i.title)

# 创建工作表：
wb.create_sheet("测试工作表")
  
# 拷贝工作表：
sheet_copy = wb.copy_worksheet(wb["测试工作表"])
sheet_copy.title = "测试工作表-复制"

# 删除指定工作表：
wb.remove(wb["测试工作表"])


#print(datetime.now().strftime('%Y%m%d%H%M%S%f'))

# 执行完修改之后，还需要持久化到本地磁盘，一般情况我们使用另存
wb.save("D:/办公/DCOS/项目进度/云平台支撑域2023年项目规划 - python.xlsx")
