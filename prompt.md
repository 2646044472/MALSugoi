我要开发一个基于MyAnimeList网站的评分进行动漫推荐的软件。
我要实现以下功能：
1. 爬虫。爬取MAL的番剧评分、标签，用户信息。
2. 实现content-based，user-based的推荐。

目前的数据有：
1. 大概三千个番剧数据，包括title,score,genres,ranked,popularity,members,favorites
2. 大概一千个用户数据，包扩用户id,这个用户对一些番剧的评分1-10.

想要实现的功能是根据新用户给出的喜欢genre或者新用户对已有的一些番剧的评分推荐番剧。
