============================
细粒度实体分类
============================

细粒度的实体分类，又叫 **Entity Typing** ，它可以将命名实体识别的结果进行更细粒度的分类，对应了OKNLP中的typing子模块。在OKNLP工具包中目前支持了基于 **BERT** 的算法，可以使用 ``oknlp.typing.get_by_name`` 来创建它。

在这篇文档中，我们主要将介绍细粒度实体分类工具的基本用法。

示例代码
=============================

>>> import oknlp
>>> model = oknlp.typing.get_by_name("bert")
>>> result = model([
...   ("我爱北京天安门", (2, 4)),
...   ("天安门上太阳升", (0, 3))
... ])
>>> result
[[('location', 0.7184367775917053), ('place', 0.8161897659301758), ('city', 0.622787356376648), ('country', 0.12566784024238586)], [('person', 0.22703033685684204), ('location', 0.20146676898002625), ('place', 0.45954859256744385)]]

输入 & 输出 说明
==============================

在细粒度实体分类任务的输入中，主要包含了两个部分：原句和实体位置。其中，实体位置是由一个左闭右开的区间来表示的。在示例代码中，第一句的实体为“北京”，第二句的实体为“天安门”。

输出为一系列可能的实体类别的概率。目前可能的实体类别如下所示：

.. list-table:: 实体列表
    :widths: 10 10 10 10 10 10 10 10 10 10
    :header-rows: 0
    
    * - person
      - group
      - organization
      - location
      - entity
      - time
      - object
      - event
      - place
      - accident
    * - actor
      - agency
      - airline
      - airplane
      - airport
      - animal
      - architect
      - army
      - art
      - artist
    * - athlete
      - attack
      - author
      - award
      - biology
      - body_part
      - bridge
      - broadcast
      - broadcast_station
      - building
    * - car
      - cemetery
      - chemistry
      - city
      - coach
      - company
      - computer
      - conflict
      - country
      - county
    * - currency
      - degree
      - department
      - director
      - disease
      - doctor
      - drug
      - education
      - election
      - engineer
    * - ethnic_group
      - facility
      - film
      - finance
      - food
      - game
      - geography
      - god
      - government
      - health
    * - heritage
      - holiday
      - hospital
      - hotel
      - institution
      - instrument
      - internet
      - island
      - language
      - law
    * - lawyer
      - league
      - legal
      - leisure
      - library
      - living_thing
      - mass_transit
      - medicine
      - military
      - mobile_phone
    * - monarch
      - mountain
      - music
      - musician
      - music_school
      - natural_disaster
      - news
      - news_agency
      - other
      - park
    * - planet
      - play
      - political_party
      - politician
      - product
      - programming_language
      - protest
      - province
      - rail
      - railway
    * - religion
      - religious_leader
      - restaurant
      - road
      - scientific_method
      - ship
      - sign
      - society
      - software
      - soldier
    * - spacecraft
      - sport
      - stage
      - stock_exchange
      - structure
      - subway
      - team
      - television_channel
      - television_network
      - television_program
    * - theater
      - title
      - train
      - transit
      - transportation
      - treatment
      - water
      - weapon
      - website
      - writing