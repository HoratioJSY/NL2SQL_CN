## Introduction

This repo is maintained for utilizing free Tesla T4 GPU in Google colab.

The Query and Table data can be found in TianChi Challenge, following sample demonstrate one query and one labeled sql data: 

```json
{
    "table_id": "4d29d0513aaa11e9b911f40f24344a08",
    "question": "二零一九年第四周大黄蜂和密室逃生这两部影片的票房总占比是多少呀",
    "sql": {
        "agg": [5],
        "cond_conn_op": 2,
        "sel": [2],
        "conds": [
            [0, 2, "大黄蜂"],
            [0, 2, "密室逃生"]
        ]
    }
}
```



The model wants to use "question" and corresponding Table data  to predict "sql". The model decouples the task of generating a whole SQL into several sub-tasks, including select-number, select-column, select-aggregation, condition-number, condition-column and so on.

The sample Table which paired with sample question can be demonstrated as:

```json
{
    "rows": [
        ["死侍2：我爱我家", 10637.3, 25.8, 5.0],
        ["白蛇：缘起", 10503.8, 25.4, 7.0],
        ["大黄蜂", 6426.6, 15.6, 6.0],
        ["密室逃生", 5841.4, 14.2, 6.0],
        ["“大”人物", 3322.9, 8.1, 5.0],
        ["家和万事惊", 635.2, 1.5, 25.0],
        ["钢铁飞龙之奥特曼崛起", 595.5, 1.4, 3.0],
        ["海王", 500.3, 1.2, 5.0],
        ["一条狗的回家路", 360.0, 0.9, 4.0],
        ["掠食城市", 356.6, 0.9, 3.0]
    ],
    "name": "Table_4d29d0513aaa11e9b911f40f24344a08",
    "title": "表3：2019年第4周（2019.01.28 - 2019.02.03）全国电影票房TOP10",
    "header": ["影片名称", "周票房（万）", "票房占比（%）", "场均人次"],
    "common": "资料来源：艺恩电影智库，光大证券研究所",
    "id": "4d29d0513aaa11e9b911f40f24344a08",
    "types": ["text", "real", "real", "real"]
}
```



The final prediction for SQL Model is that:

```sql
SELECT SUM(col_3) FROM Table_4d29d0513aaa11e9b911f40f24344a08 WHERE (col_1 == '大黄蜂' and col_1 == '密室逃生')
```

More details about model and data can be found: (Blog)[http://jiangsiyuan.com/]