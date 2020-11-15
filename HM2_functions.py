#functions
import pandas as pd
import seaborn 
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick 
from matplotlib.ticker import PercentFormatter

########################################################################################################################
################################################### IMPORT FILES #######################################################
########################################################################################################################

#place in l the paths of the Data Frames
l = ['/Users/paolaantonicoli/2019-Oct.csv','/Users/paolaantonicoli/2019-Nov.csv']



def conversione_ID_code(category, lista = l):
    '''
    category_id in category_code converter and vv.
    :param l: list of paths
    :param category: category_id / category_code
    :return: category_code /category_id
    '''
    cols = ['category_id','category_code']
    out = pd.DataFrame({},columns = cols)
    for path in lista:
        for chunk in pd.read_csv(path, header = 'infer', usecols= cols, chunksize = 10000):
            chunk = chunk.drop_duplicates()
            out = pd.concat([chunk,out])
            del chunk
        out = out.drop_duplicates()
        if str(category).isdigit():
            out = out.loc[out['category_id']==category]
            return out['category_code'].iloc[0]
                                                 
        else:
            out = out.loc[out['category_code']==category]
            return out['category_id'].iloc[0]


        

#-----------------------------------------------[RECURRENT FUNCTIONS]---------------------------------------------------
def just_purchase(df):
    '''
    select only the rows with purchase

    :param df: a dataframe
    :return: a df with only rows with event_type == 'purchase'
    '''
    return df.loc[df['event_type'] == 'purchase']

def just_cart(df):
    '''
    select only the rows with cart

    :param df: a dataframe
    :return: a df with only rows with event_type == 'purchase'
    '''
    return df.loc[df['event_type'] == 'cart']

def just_view(df):
    '''
    select only the rows with view
    :param df: a dataframe
    :return: a df with only rows with event_type == 'view'
    '''
    return df.loc[df['event_type'] == 'view']

###MODIFICA
def get_path_mounth(path):
    first_row = pd.read_csv(path, header = 'infer', usecols= ['event_time'], nrows =1,parse_dates=['event_time'],date_parser=pd.to_datetime)
    return first_row.iloc[-1].dt.month_name()[0]



#-----------------------------------------------[INTRODUCTION]---------------------------------------------------

###MODIFICA
def info_list(l):
    for path in l:
        mese = get_path_mounth(path)
        df = pd.read_csv(path, header = 'infer')
        print('\n' + mese.center(45, '-') + '\n')
        print(info(df))

def info(df):
    return df.info(verbose=True, null_counts=True)


########################################################################################################################
###################################################  RQ 1 ##############################################################
########################################################################################################################
# us 1: view view purch view
# us 2 : view cart view
# us 3: view cart cart
#___________________________________________________RQ 1.1 _____________________________________________________________

###MODIFICA
def main_event_type(l):
    out = pd.DataFrame({})
    for path in l:
        df = pd.read_csv(path, usecols = ['event_type'], header = 'infer')
        df = df.drop_duplicates()
        out = pd.concat([out, df])
    out = out.drop_duplicates()
    
    return out['event_type'].reset_index(drop='True')

    

def RQ_1_1(l):
    '''
    :param l: list of paths
    :return: a df containing the number of times an event occours in the dfs and the avg value of that for each user_session
    '''
    cols = ['user_session','event_type']
    out = pd.DataFrame({}, columns=['user_session', 'event_type','count_event_type'])
    for path in l:
        for chunk in pd.read_csv(path, header='infer', usecols=cols,chunksize= 1000000):
            chunk = chunk.groupby(['user_session','event_type']).event_type.count().to_frame('count_event_type').reset_index()
            out = pd.concat([out,chunk])
            del chunk
    # number of user session counted once
    us_card = len(out['user_session'].unique())

    #sum for each event
    out = out.groupby('event_type').count_event_type.sum().to_frame('number_of_events').reset_index()

    out['avg_number_of_times']= out['number_of_events']/us_card
    return(out)

def plot_RQ_1_1(result): #result = result of RQ_1_1
    pie, ax = plt.subplots(figsize=[12,6])
    labels = result["event_type"]
    sizes=result["avg_number_of_times"]
    colors=['#d5c0b2',"#865067",'#bc918d']
    plt.pie(sizes, autopct="%.1f%%", explode=[0.1]*3, colors=colors, labels=labels, pctdistance=0.5,shadow=True, startangle=45)
    plt.title("the average number of each operation", fontsize=20)


#___________________________________________________RQ 1.2 _____________________________________________________________
#specifichiamo che qui non abbiamo volutamente utilizzato il parse date time perchè il formato è già ordinato

def views_berfore_purchasing(l):
    '''
    :param l: list of paths
    :return: a df with for each row the user_id and the product_id of a product purchesed by the user, the time of purchasing and the previous times of viewing  
    '''
    cols = ['user_id','product_id','event_type']
    out = pd.DataFrame({})

    #create a dict : the keys are the product_id and the user_id, the contenent is a df with the viwe/purchase time 
   
    view = pd.DataFrame({})
    cart = pd.DataFrame({})
    for path in l:
        for chunk in pd.read_csv(path, header='infer', usecols=cols,chunksize= 1000000 ):
            chunk_cart = just_cart(chunk)
            del chunk_cart['event_type']
            cart = pd.concat([cart,chunk_cart])
            del chunk_cart
            
            chunk_view = just_view(chunk)
            chunk_view = chunk_view.groupby(['user_id','product_id']).event_type.count().to_frame('number_views').reset_index()
            view = pd.concat([view,chunk_view])
            del chunk_view
    
    view = view.groupby(['user_id','product_id']).number_views.sum().to_frame('number_views').reset_index()
    out = pd.merge(view,cart, how = 'inner') 
    return out

        
#Once we computed how many times the userX views the product_i (e.g. p1 = 3 , p2 = 4) we compute the mean(3.5) then we compute the mean for each user 

def user_views_before_purchasing(df):
    '''
    :param l: df parsed by views_before_purchasing()
    :return: a df with the avarege times a user views a product befor buying it 
    '''
    df = df.groupby(['user_id']).number_views.mean().to_frame('number_of_views').reset_index()
    return df

def RQ_1_2(l):
    df1 = views_berfore_purchasing(l)
    df2 = user_views_before_purchasing(df1)
    return df2.number_of_views.mean()

#___________________________________________________RQ 1.3 _____________________________________________________________

# we interpret this prob as the parameter of a binomial, where X= 1 if product added to the cart by a user is purchased, X=0 else

##MODIFICA DEL DEL

def RQ_1_3(l):
    '''
    :param l: list of paths
    :return: the probability of a product added to cart to be effectively bought
    '''
    
    cols = ['user_id','event_type','product_id']
    carted = pd.DataFrame({})
    purchased = pd.DataFrame({})
    for path in l:
        for chunk in pd.read_csv(path, header='infer', usecols=cols,chunksize= 1000000):
            
            #create a df with the users 'user_id' that added the product 'product_id', once for every time the product has been added to cart
            chunk_carted = just_cart(chunk)
            del chunk_carted['event_type']
            chunk_carted = chunk_carted.drop_duplicates()
            carted = pd.concat([carted, chunk_carted])
            carted = carted.drop_duplicates()
            del chunk_carted

            #create a df with the users 'user_id' that purchased the product 'product_id', once for every time the product has been purchased
            chunk_purchased = just_purchase(chunk)
            del chunk_purchased['event_type']
            chunk_purchased = chunk_purchased.drop_duplicates()
            purchased = pd.concat([purchased, chunk_purchased])
            purchased = purchased.drop_duplicates()
            del chunk_purchased
       
            
            del chunk
            
    #merge to get the intersection of products that are both purchased and added to the cart
    prod = pd.merge(carted,purchased, how ='inner')
    
    return len(prod)/len(carted)

#___________________________________________________RQ 1.4 _____________________________________________________________

def time_difference(c, p):
    '''
    Time difference between c and p

    :param c: time in str format
    :param p: time in str format
    :return: time difference between p and c
    '''
    if c < p:
        c = pd.to_datetime(c)
        p = pd.to_datetime(p)
        return p - c
    else:
        return 'NaN'


def cart_removed(l):
    '''
    :param l: list of paths
    :return: a df with the time that occours between the first view time and the add to cart ##MODIFICA
    '''
    cols = ['event_time', 'user_session', 'event_type', 'product_id']
    carted = pd.DataFrame({}, columns=cols)
    purchased = pd.DataFrame({}, columns=cols)
    for path in l:
        for chunk in pd.read_csv(path, header='infer', usecols=cols, chunksize=1000000):
            carted_chunk = just_cart(chunk)
            carted = pd.concat([carted, carted_chunk])
            del carted_chunk
            purchased_chunk = just_purchase(chunk)
            purchased_chunk = purchased_chunk.drop_duplicates(subset=['user_session', 'event_type', 'product_id'])
            purchased = pd.concat([purchased, purchased_chunk])
            del purchased_chunk
    # we want just the first purchasing of a product for each user_session
    purchased = purchased.drop_duplicates(subset=['user_session', 'event_type', 'product_id'])

    # renaming for the merge
    purchased = purchased.rename(columns={'event_time': 'purchasing_time'}, inplace=False)
    del purchased['event_type']

    # renaming for the merge
    carted = carted.rename(columns={'event_time': 'carting_time'}, inplace=False)
    del carted['event_type']

    # we want on the same row the time of purchasing and the time of addition to cart
    comparing = purchased.merge(carted, how='inner', on=['user_session', 'product_id'])

    # difference between carting and purchasing time
    comparing['time_diff'] = comparing.apply(lambda x: time_difference(c=x['carting_time'], p=x['purchasing_time']),
                                             axis=1)
    comparing = comparing.dropna()
    return comparing

def RQ_1_4(l):
    df = cart_removed(l)
    return df.time_diff.mean()

### scrivere in formato stringa l'output

#___________________________________________________RQ 1.5 _____________________________________________________________


def time_first_view_and_pur_cart(l):
    '''

    :param l: list of paths
    :return: a df with the time that occours between the first time the user views a product and the first between time he adds it to cart or purchases it
    '''
    cols = ['event_time', 'user_id', 'event_type', 'product_id']
    carted = pd.DataFrame({}, columns=cols)
    purchased = pd.DataFrame({}, columns=cols)
    view = pd.DataFrame({}, columns=cols)

    for path in l:
        for chunk in pd.read_csv(path, header='infer', usecols=cols, chunksize=3000000):
            
            view_chunk = just_view(chunk)
            #view_chunk = view_chunk.drop_duplicates(subset=['user_id', 'event_type', 'product_id']) MODIFICA
            view = pd.concat([view, view_chunk])
            del view_chunk
            
            carted_chunk = just_cart(chunk)
            carted = pd.concat([carted, carted_chunk]).drop_duplicates(subset=['user_id', 'event_type', 'product_id'])
            del carted_chunk
            
            purchased_chunk = just_purchase(chunk)
            #purchased_chunk = purchased_chunk.drop_duplicates(subset=['user_id', 'event_type', 'product_id']) MODIFICA
            purchased = pd.concat([purchased, purchased_chunk])
            del purchased_chunk
            
            del chunk
        
    # we want just the first purchasing of a product for each user_session
    view = view.drop_duplicates(subset=['user_id', 'event_type', 'product_id'])
    carted = carted.drop_duplicates(subset=['user_id', 'event_type', 'product_id'])
    purchased = purchased.drop_duplicates(subset=['user_id', 'event_type', 'product_id'])
    
    #first beteen carted or purchased
    cart_pur = pd.concat([carted, purchased])
    
    del carted
    del purchased
    
    cart_pur = cart_pur.sort_values('event_time')
    cart_pur = cart_pur.drop_duplicates(subset=['user_id', 'event_type', 'product_id'])

    # renaming for the merge
    #cart_pur = cart_pur.rename(columns={'event_time': 'carting_purchasing_time'}, inplace=False)
    cart_pur.rename(columns={'event_time': 'carting_purchasing_time'}, inplace=True) 
    del cart_pur['event_type']

    # renaming for the merge
    #view = view.rename(columns={'event_time': 'view_time'}, inplace=False)
    view.rename(columns={'event_time': 'view_time'}, inplace=True) #MODIFICA
    del view['event_type']

    # we want on the same row the time of first view and the time of addition to cart/purchasing
    comparing = cart_pur.merge(view, how='inner', on=['user_id', 'product_id'])

    # difference between carting and purchasing time
    comparing = comparing.loc[comparing['view_time']<comparing['carting_purchasing_time']]  
    
    comparing['time_diff'] = comparing.apply(lambda x: time_difference(c=x['view_time'], p=x['carting_purchasing_time']),axis=1)


    return comparing


def RQ_1_5(l):
    df = time_first_view_and_pur_cart(l)
    return df.time_diff.mean()



########################################################################################################################
###################################################  RQ 2 ##############################################################
#########################################################################################################################
#
# What are the categories of the most trending products overall? For each month visualize this information through a
# plot showing the number of sold products per category.


#___________________________________________________RQ 2.1 _____________________________________________________________

#ASSUMPTIONS = SUBCATEGORY := THE ONE AFTER THE DOTS (IG : furniture.bedroom.bed	CAT = furniture , SUBCAT = bedroom.bed)

def split_category(df):
   ''' given a df ,returns one with prefix and suffix of the cathegory, splitted by the first dot
   :param df: df with category_code label
   :return: df with the column category_code splitted in 'cat' and 'subcat'
   '''
   new =  df.category_code.str.split('.',1,expand=True)
   df['cat'] = new[0]
   df['sub_cat'] = new[1]
   return df[['cat','sub_cat']]

def subcategory_visits(l):
    '''

    :param l: list of paths with CSV files
    :return: a df with 'sub_cat' and 'number_visits' = number of visits for each subcategory
    '''

    cols = ['category_code']
    out = pd.DataFrame({},columns = ['sub_cat','number_of_visits','category_code','cat'])
    for path in l:
        for chunk in pd.read_csv(path, header = 'infer', usecols= cols, chunksize = 1000000):

            # delete the NaN values
            chunk = chunk.dropna()

            # splits in Cat and Sub Cat
            chunk = split_category(chunk)

            # Number of visits for every chunk
            chunk= chunk.groupby('sub_cat').cat.count().to_frame('number_of_visits').reset_index()
            out= pd.concat([out,chunk])

        # sums for every numb of visit found per chunks
    return out.groupby('sub_cat').number_of_visits.sum().to_frame('number_visits').reset_index()


def top_n(df, criteria='number_visits', n=-1):
    ''' returns the top-n values of the df sorted by 'criteria' lable
    :param df: df with sub_cat and n_visits for each sub_cat
    :param criteria: the column with respect to wich we sort
    :param n: Top n visited, default all
    :return: the n-top visited sub cat and number of visits, if n missing, all of them
    '''
    if n == -1:
        n = len(df)

    out = df.sort_values(criteria, ascending=False).head(n)
    return out

def RQ_2_1(l):
    df = subcategory_visits(l)
    return top_n(df)

def plot_RQ_2_1(result): #result = result of RQ_2_1
    result=result.sort_values(['number_visits'],ascending=False).head(10)
    seaborn.set_style("whitegrid")
    ax=seaborn.catplot(x="sub_cat",y="number_visits", kind="bar",palette="ch:.25", data=result,height=6, aspect = 2)
    plt.title('The first 10 visited subcategories')
    ax.set_xticklabels(rotation=50)
    ax.set(xlabel='Subcategory', ylabel='Number of visits')
    plt.show()

#___________________________________________________RQ 2.2 _____________________________________________________________

#ASSUMPTIONS : Category = Cathegory_ID

def number_of_purchases_for_category(l):
    '''

    :param l: list of paths associated to dfs
    :return: a df with the number of purchases for each product for each category
    '''
    cols = ['category_id', 'category_code', 'event_type', 'product_id']
    out = pd.DataFrame({}, columns=cols + ['number_of_purchases'])
    for path in l:
        for chunk in pd.read_csv(path, header='infer', usecols=cols, chunksize=1000000):
            chunk = just_purchase(chunk)
            chunk = chunk.groupby(['category_id', 'category_code', 'product_id']).event_type.count().to_frame(
                'number_of_purchases').reset_index()
            out = pd.concat([out, chunk])
            del chunk

    return out.groupby(['category_id', 'category_code', 'product_id']).number_of_purchases.sum().to_frame(
        'number_purchases').reset_index()


def top_10_prod(df):
    '''

    :param df: a df with the number of purchases for each product for each category
    :return: a df with the top-10 purchased products for each category
    '''
    out = pd.DataFrame({}, columns=['category_id', 'category_code', 'product_id', 'number_purchases'])
    for index, g in df.groupby('category_id'):
        r = top_n(g, 'number_purchases', 10)
        out = pd.concat([out, r])
        del r
    return (out)

def RQ_2_2(l):
    df = number_of_purchases_for_category(l)
    return top_10_prod(df)

def RQ_2_2_format(result): #result = result of RQ_2_2
    result = result.sort_values(['category_code','number_purchases'],ascending = False)
    out = pd.DataFrame({})
    for name, group in result.groupby('category_id'):
        s = pd.Series(list(group['product_id']),name='category_code:  ' + str(name))
        out = pd.concat([out,s],axis = 1)
    return out

########################################################################################################################
################################################### [ RQ 3 ] ###########################################################
########################################################################################################################
#For each category, what’s the brand whose prices are higher on average?

#___________________________________________________RQ 3.1 _____________________________________________________________


def avgPrice_br_cat(l):
    '''
    :param l: list of path
    :return: a df with the avg prices by brand of each product for each category
    '''
    cols = ['brand', 'category_id', 'product_id', 'price']
    out = pd.DataFrame({}, columns=cols)
    for path in l:
        for chunk in pd.read_csv(path, header='infer', usecols=cols, chunksize=1000000):
            # at every chunk I sign the sum of the prices and the number of different product
            chunk = chunk.drop_duplicates()
            out = pd.concat([out, chunk])
            del chunk
    out = out.drop_duplicates()
    out = out.groupby(['category_id', 'brand']).price.mean().to_frame('mean_values').reset_index()

    return out[['category_id', 'brand', 'mean_values']]

def RQ3_1(l):
    '''
    This function asks in input a category and returns the avg price by brand of the products associated to that category
    :param l: list of paths of CSV files
    :return: avg price by brand of the products associated to that category
    '''

    name = input("Enter Category (ID or Code) ")
    if not str(name).isdigit():
        name = conversione_ID_code(l,name)
    df = avgPrice_br_cat(l)

    return df.loc[df['category_ID']== name ]



def plot_RQ3_1(result): #result = result of RQ_3_1
    seaborn.set_style("whitegrid")
    palette=seaborn.color_palette("pastel")
    ax=seaborn.catplot(x="brand",y="mean_values", kind="bar",palette=palette, data=result,height=6, aspect = 2)
    plt.title('The average price of the products sold by the brand.')
    ax.set_xticklabels(rotation=50)
    ax.set(xlabel='Brand', ylabel='Average price')
    plt.show()
#___________________________________________________RQ 3.2 _____________________________________________________________
def highest_avg_price(df):
    '''

    :param df: data frame with colums 'mean_values' of mean price of brands for each category, and 'category_id'
    :return: a df with the brand with the highest avg price for each category id
    '''
    df = df.sort_values('mean_values', ascending=False)
    df = df.drop_duplicates('category_id')
    return df

def RQ3_2(l): 
    df = avgPrice_br_cat(l)
    df = highest_avg_price(df).head(10)                                
    return df

def plot_RQ3_2(result):
    seaborn.set_style("whitegrid")
    #palette=seaborn.color_palette("pastel")
    ax=seaborn.catplot(x="category_id",y="mean_values",hue="brand", kind="bar",palette="ch:.25", data=result,height=6, aspect = 3 ,order = result.sort_values('mean_values',ascending = False).category_id,dodge=False)
    plt.title('The brand with the highest avg price')
    ax.set_xticklabels(rotation=90)
    ax.set(xlabel='Brand', ylabel='Average price')
    plt.show()

    
########################################################################################################################
################################################### [ RQ 4 ] ###########################################################
########################################################################################################################
#How much does each brand earn per month? Write a function that given the name of a brand in input returns,
# for each month, its profit. Is the average price of products of different brands significantly different?
#Using the function you just created, find the top 3 brands that have suffered the biggest losses in earnings
# between one month and the next, specifing bothe the loss percentage and the 2 months (e.g., brand_1 lost 20%
# between march and april).


def sum_of_earns(df,on = 'price'):
    ''' this function returns the sum of the values 'on' (default price) for each brand
    we use it to avoid using the same function in and out the chunk analysis in earns_for_month()
    :param df: df in input
    :param on:
    :return: sum of the values 'on' (default price) for each brand
    '''
    df =  df.groupby('brand')[on].sum().to_frame('earnings').reset_index()
    return df[['brand','earnings']]

def earns_for_month(l):
    '''

    :param l: list of paths
    :return: list with df ,one for each month, where for each brand are returned the total earns for that month
    '''
    cols = ['brand','event_type','price']
    list_of_earns = []
    for path in l:
        mese = get_path_mounth(path) 
        out_mensile = pd.DataFrame({},columns = cols+['mese'])
        for chunk in pd.read_csv(path, header = 'infer', usecols= cols, chunksize = 1000000):
            chunk = just_purchase(chunk)
            chunk = sum_of_earns(chunk)
            out_mensile = pd.concat([out_mensile,chunk])
            del chunk
        out_mensile = sum_of_earns(out_mensile, on = 'earnings')
        out_mensile = out_mensile.rename(columns = {'earnings':'Earnings of '+str(mese)}, inplace = False)
        list_of_earns += [out_mensile]
        del out_mensile
    return list_of_earns


def time_comparing(list_of_earns):
    '''
    used to merge all the dfs in the output list of earns_for_months()
    :param list_of_earns: list output of earns_for_months()
    :return: a df where for each brand(row) are reported the earnings for each month(column)
    '''
    df1 = list_of_earns.pop()
    for df in list_of_earns:
        df1 = df1.merge(df,how='outer',on = 'brand')
    return df1.fillna(0)

def asking_brand_input(l):
    '''
    this function asks in input a brand and returns the earnings of that brand for each month
    :param l: list of path
    :return: the earnings of that brand for each month
    '''
    brand = input()
    list_of_earns = earns_for_month(l)
    df = time_comparing(list_of_earns)
    return df.loc[df['brand']==brand]

def RQ_4_1(l):
    return asking_brand_input(l)[['brand','Earnings of October','Earnings of November','Earnings of December','Earnings of January','Earnings of February','Earnings of March','Earnings of April']]


def percentage_loss(i1,i2):
    '''
    :param i1: earnings of the first month
    :param i2: earnings of the second month
    :return: returns the percentage value of difference between i1 and i2, if i1 == 0 then returns 0
    '''
    if i1 == 0:
        return 0
    else:
        return ((i2-i1)/i1)*100

def perc_loss_between_months(df):
    '''

    :param df: ad df processed by time_comparing()
    :return: a df where is indicated for each brand(row) the percentage loss(without %) and the couple
    of months respect to which we consider the loss 'mm1-mm2'
    '''
    l= ['brand','Earnings of October','Earnings of November','Earnings of December','Earnings of January','Earnings of February','Earnings of March','Earnings of April']    
    loss = pd.DataFrame({},columns = ['brand','loss','months'])
    for i in range(1, len(l)-1):
        loss_i = pd.DataFrame({},columns = ['brand','loss','months'])
        prec_month = l[i]
        month = l[i+1]
        loss_i['brand'] = df['brand']
        loss_i['loss'] = df.apply(lambda x: percentage_loss(i1 = x[month], i2 = x[prec_month]), axis=1) #percentage loss
        loss_i['months'] = prec_month[12:]+' - '+month[12:]  #create a column that spec. the months
        loss = pd.concat([loss,loss_i])
        del loss_i
    return loss

def biggest_loss(df):
    '''
    a df processed by prec_loss_between_months
    we sort it according to the loss
    :param df: a df processed by prec_loss_between_months
    :return: top 3 brands that suffered a loss and the couple of month in wich that occured
    '''
    df = df.sort_values('loss',ascending = False)
    out = df.head(3)
    return out

def RQ_4_2(l):
    list_of_earns = earns_for_month(l)
    merged = time_comparing(list_of_earns)
    df = perc_loss_between_months(merged)
    df = biggest_loss(df)
    return df

def delite_earnings(s):
    '''
    Returns the str without the suffix 'Earnings of'
    :param s: str whith format 'Earning of 'mounth''
    :return: 'mounth'
    '''
    print(s)
    return s[12:]



def top_3_losses_earnings(df): 
    '''
    Returns the values of earnings for each month of the brands that suffered the biggest loss
    :param l: df out of RQ4_2
    :input : list_of_top
    :return: the values of earnings for each month of the brands that suffered the biggest loss
    '''
    list_of_top=list(df['brand'])
    out=pd.DataFrame({})
    for brand in list_of_top:
        df=RQ_4_1(brand)
        out=pd.concat([out,df])
    out =pd.melt(out ,id_vars=["brand"], 
        var_name="mounth", 
        value_name="earnings")
    out['mounth'] = out.apply(lambda x : delite_earnings(x['mounth']),axis = 1)
    return out.sort_values(['brand','mounth'])


def plot_RQ_4_2(out): 
    '''
    Returns a plot with the profit trend of the brand during the interested period 
    :input: result of def top_3_losses_earnings(l)
    :return: a plot with the profit trend of the brand during the interested period 
    '''
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "Dec"]
    out['mounth'] = pd.Categorical(out['mounth'], categories=months, ordered=True)
    out.sort_values(by="mounth",inplace=True)
    ax=seaborn.relplot(x="mounth", y="earnings",hue="brand",kind="line",data=out,palette=["blue","orange","salmon"],height=6, aspect=2, facet_kws=dict(sharex=False))
    plt.xlabel('Month', fontsize=18)
    plt.ylabel('Value obtained', fontsize=16)
    plt.title('The profit trend of the brand')
    plt.show()

def string_biggest_loss(out): 
    '''
    Returns,like a string,the name, loss and period's loss of three biggest loss brands 
    :input: result of def RQ_4_2(l)
    :return: the name, loss and period's loss of three biggest loss brands
    '''
    out['loss'] = out['loss'].div(100).round(2)
    out['mounth']= out['months'].replace('-',' and ')
    for index, row in out.iterrows():
        string = ''
        for x in range(len(row)):
            string += '%s, ' % row[x]
        string=string.split(",")
        string[0]='"'+ string[0] + '"'
        print("One of the biggest loss brand is %s" %string[0].capitalize(), "with a loss %s" %string[1],"%", "during the period beetween%s" %string[2])

### scrivere una funzione che renda in formato stringa la richiesta di scrivere i 3 output di biggest_loss

########################################################################################################################
################################################### [ RQ 5 ] ###########################################################
########################################################################################################################

#In what part of the day is your store most visited?
# Knowing which days of the week or even which hours of the day shoppers are
# likely to visit your online store and make a purchase may help you improve your strategies

#Create a plot that for each day of the week show the hourly average of visitors your store has.


def get_week_day_hour(df):
    '''
    used to extract the time infos
    :param df: a df with the column 'event_time'
    :return: a df with the columns 'week', 'day', 'hour'
    '''
    df['week'] = df['event_time'].dt.week
    df['day'] = df['event_time'].dt.weekday
    df['hour'] = df['event_time'].dt.hour
    del df['event_time']
    return df


def get_number_of_visitors(df):
    '''
    :param df: a df parsed by get_week_day_hour()
    :return: return the number of visitors for each week, day, hour
    '''
    df = get_week_day_hour(df)
    df = df.drop_duplicates()  # we are interested only in counting the number of users that access to the site once in the site
    df = df.groupby(['week', 'day', 'hour']).user_id.count().to_frame('number_of_visitors').reset_index()
    return df


def avg_visits_for_hour_for_dayweek(l):
    '''

    :param l: list of paths
    :return: a df with the avg visitors for day and hour
    '''
    cols = ['user_id', 'event_time']
    out = pd.DataFrame({}, columns=['week', 'day', 'hour', 'mounth', 'number_of_visitors'])
    for path in l:
        first_row = pd.read_csv(path, header='infer', usecols=['event_time'], nrows=1, parse_dates=['event_time'],
                                date_parser=pd.to_datetime)  # read first row just for the monty
        mese = first_row.iloc[-1].dt.month_name()[0]  # find the month of each df
        for chunk in pd.read_csv(path, header='infer', usecols=cols, chunksize=1000000, parse_dates=['event_time'],
                                 date_parser=pd.to_datetime):
            chunk = get_number_of_visitors(chunk)
            chunk['mounth'] = mese
            out = pd.concat([out, chunk])
            del chunk
    number_visit = out.groupby(['mounth', 'week', 'day', 'hour']).number_of_visitors.sum().to_frame(
        'number_of_visitors').reset_index()
    number_visit_day_hour = number_visit.groupby(['day', 'hour']).number_of_visitors.mean().to_frame(
        'avg_days').reset_index()
    del number_visit
    return number_visit_day_hour

def RQ_5(l):
    return avg_visits_for_hour_for_dayweek(l)

def day_of_week(n):
    l = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    return l[n]

def plot_RQ_5(df):
    seaborn.set_style("whitegrid")
    for i in range(7):
        df1 = df.loc[df['day']==i]
        ax=seaborn.catplot(x="hour",y="avg_days", kind="point",palette="ch:.25", data=df1,height=4.5, aspect = 1.5)
        plt.title('Average number of visitors per hour throughout ' + day_of_week(i))
        ax.set_xticklabels(rotation=50)
        ax.set(xlabel='Hour', ylabel='Visitors_average')
        del df1
    plt.show()
########################################################################################################################
################################################### [ RQ 6 ] ###########################################################
########################################################################################################################


def prod_views_purchases(l, on = 'product_id'):
    '''
    get the number of views , purchases for each product
    :param l: list of paths
    :return: df with the number of views , purchases for each products
    '''
    cols = ["event_type", on]
    out = pd.DataFrame({}, columns=cols + ["number_of_views", "number_of_purchases"])
    for path in l:
        for chunk in pd.read_csv(path, header='infer', usecols=cols, chunksize=1000000):
            # number of views for each product
            chunk_purchased = just_purchase(chunk)
            chunk_purchased = chunk_purchased.groupby(on).event_type.count().to_frame(
                'number_of_purchases').reset_index()
            # number of purchases for each product
            chunk_view = just_view(chunk)
            chunk_view = chunk_view.groupby(on).event_type.count().to_frame('number_of_views').reset_index()

            # merge to have on the same row 'product_id, n.views, n.purchases'
            final = chunk_view.merge(chunk_purchased, how="outer", on=on)
            out = pd.concat([out, final])

            del chunk_view
            del chunk_purchased

    # sum for each product_id the n.views and n.purchases
    out = out.groupby(on).agg(
        tot_number_of_views=pd.NamedAgg(column="number_of_views", aggfunc="sum"),
        tot_number_of_purchases=pd.NamedAgg(column="number_of_purchases", aggfunc="sum")).reset_index()

    return out


def conversion_rate(p, v):
    '''
    conversion rate of a product with p = number of purchases and v = number of views
    :param p: number of purchases
    :param v: number of views
    :return: conversion rate
    '''
    if v == 0:
        return 0
    else:
        return p / v

def conversion_rate_for_prod(l,on='product_id'):
    df = prod_views_purchases(l,on)
    df['conversion_rate'] = df.apply(lambda x: conversion_rate(p = x['tot_number_of_purchases'], v = x['tot_number_of_views']), axis=1)
    return df[[on,'conversion_rate']]

def RQ_6_1(df): #df = df output of prod_views_purchases() 
    p = df['tot_number_of_purchases'].sum()
    v = df['tot_number_of_views'].sum()
    return conversion_rate(p,v)

def RQ_6_2(df): #df = df output of conversion_rate_for_prod(l,on = 'category_id')
    return df.sort_values('conversion_rate',ascending = False)

def plot_RQ_6_2(result): #df = df output of RQ_6_2
    seaborn.set_style("whitegrid")
    #palette=seaborn.color_palette("pastel")
    ax=seaborn.catplot(x="category_id",y="conversion_rate", kind="bar",palette="ch:.25", data=result,height=6, aspect = 2,order=result.sort_values("conversion_rate",ascending=False).category_id)
    plt.title('The Conversion rate for category.')
    ax.set_xticklabels(rotation=90)
    ax.set(xlabel='Category ID', ylabel='Convesion rate')
    plt.show()
    

def plot2_RQ_6_2_pfc(out):
    seaborn.set_style("whitegrid")
    ax=seaborn.catplot(x="category_id",y="tot_number_of_purchases", kind="bar",palette="ch:.25", data=out,height=6, aspect = 2,order=out.sort_values("tot_number_of_purchases",ascending=False).category_id)
    plt.title('Purchases for each category')
    ax.set_xticklabels(rotation=90)
    ax.set(xlabel='Category ID', ylabel='Number of purchases')
    plt.show()

########################################################################################################################
################################################### [ RQ 7 ] ###########################################################
########################################################################################################################


# The Pareto principle states that for many outcomes roughly 80% of consequences come from 20% of the causes.
# Also known as 80/20 rule, in e-commerce simply means that most of your business, around 80%,
# likely comes from about 20% of your customers.
# Prove that the pareto principle applies to your store.



def cumulative_purchases(l):
    '''
    create a df with the cumulative values of the purchases of the users
    :param l: list of paths
    :return: a df with the total and cumulative values of the purchases of the users
    '''
    cols = ['user_id', 'price','event_type']
    out = pd.DataFrame({}, columns=[]+['chunk_user_purchase'])
    for path in l:
        for chunk in pd.read_csv(path, header='infer', usecols=cols, chunksize=100000):
            chunk = just_purchase(chunk)
            chunk = chunk.groupby('user_id').price.sum().to_frame('chunk_user_purchase').reset_index()
            out = pd.concat([chunk, out])
            del chunk
    out = out.groupby('user_id').chunk_user_purchase.sum().to_frame('user_purchase').reset_index()
    out = out.sort_values('user_purchase',ascending = False)
    out['cumulative_purchases'] = out['user_purchase'].cumsum()
    return out[['user_id','user_purchase','cumulative_purchases']]

def classes(df):
    '''

    :param df: a df parsed by cumulative_purchases()
    :return: a df with the users divided in classes A and B if they contributed at the 80% or 20% or cumulative purchases
    '''
    total_earnings = df['cumulative_purchases'].iloc[-1]
    print(total_earnings)
    df['percentage_earnings'] = (df['cumulative_purchases']/total_earnings)*100
    t_A = 0.8*total_earnings #the threshold for being in class A
    df['class'] = 'B'
    df.loc[df['cumulative_purchases']<= t_A,'class']='A'
    return df

def RQ_7(l):
    return classes(cumulative_purchases(l))

def check_pareto(df):
    '''

    :param df: df divided in classes A and B
    :return: a str that checks Pareto's principle
    '''
    return print('number of class_A users : ' + str(len(df.loc[df['class']=='A']))+"\n" +'number of total users : ' + str(len(df)) + '\n'+'number of class_A users according Pareto :  ' + str(int(0.2*len(df))))

def create_pareto_chart(result):
    result=result.reset_index()
    l=[i for i in range(0,len(result),100)]
    result=result.iloc[l]
    result.index=list(map(str,list(result["user_id"])))
    x = result.index
    y = result["cumulative_purchases"]
    mask1 = result["class"] == "A"
    mask2 = result["class"] == "B"
    labels=["A","B"]
    plt.figure(figsize=(12,6))
    plt.bar(x[mask1], y[mask1], color = '#865067')
    plt.bar(x[mask2], y[mask2], color = '#bc918d')
    plt.legend(labels, title="Class", frameon=False)
    plt.show()

