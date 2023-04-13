# Benchmarking of GPT-X models (base model w/few shot) using PoT output on Finance QnA Datasets

## Results:
|       | dv3   | turbo35 | gpt4  |  
|-------|-------|---------|-------|  
| ConvFinQA |62.5 | 67.2 | - |  
| FinQA | - |61.2 | - |  
| AQuA  | - |- | 72.8 |
| tatqa | - |- | - |  

## Program of Thought Example:

## Prompt
<code>
<i>Read the following text and table, and then write code to answer a question:</i>
five-year performance comparison 2013 the following graph provides an indicator of cumulative total shareholder returns for the corporation as compared to the peer group index ( described above ) , the dow jones , and the s&p 500 . the graph assumes that the value of the investment in the common stock of union pacific corporation and each index was $ 100 on december 31 , 2005 and that all dividends were reinvested . purchases of equity securities 2013 during 2010 , we repurchased 17556522 shares of our common stock at an average price of $ 75.51 . the following table presents common stock repurchases during each month for the fourth quarter of 2010 : period total number of shares purchased [a] average price paid per share total number of shares purchased as part of a publicly announced plan or program [b] maximum number of shares that may yet be purchased under the plan or program [b] . [a] total number of shares purchased during the quarter includes approximately 563220 shares delivered or attested to upc by employees to pay stock option exercise prices , satisfy excess tax withholding obligations for stock option exercises or vesting of retention units , and pay withholding obligations for vesting of retention shares . [b] on may 1 , 2008 , our board of directors authorized us to repurchase up to 40 million shares of our common stock through march 31 , 2011 . we may make these repurchases on the open market or through other transactions . our management has sole discretion with respect to determining the timing and amount of these transactions . on february 3 , 2011 , our board of directors authorized us to repurchase up to 40 million additional shares of our common stock under a new program effective from april 1 , 2011 through march 31 , 2014. .
period | total number ofsharespurchased [a] | averageprice paidper share | total number of sharespurchased as part of apublicly announced planor program [b] | maximum number ofshares that may yetbe purchased under the planor program [b]
oct . 1 through oct . 31 | 725450 | 84.65 | 519554 | 17917736
nov . 1 through nov . 30 | 1205260 | 89.92 | 1106042 | 16811694
dec . 1 through dec . 31 | 1133106 | 92.59 | 875000 | 15936694
total | 3063816 | $ 89.66 | 2500596 | n/a
<b>Question: how much more was spent on shares in nov 2010 than dec 2010?</b>
</code>

## Completion
<code>
   #Python
shares_purchased_nov_2010 = 1205260
shares_purchased_dec_2010 = 1133106
difference_in_shares_purchased = shares_purchased_nov_2010 - shares_purchased_dec_2010
average_price_paid_per_share = 89.66
amount_spent_on_shares = difference_in_shares_purchased * average_price_paid_per_share
<b>ans = amount_spent_on_shares</b>
    </code>



Credit to @wenhuchen - https://github.com/wenhuchen/Program-of-Thoughts