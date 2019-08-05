"""
multiple version
Sadly, the API for trustpilot is closed and only available for enterprise users
the API has access to the following very useful data:
Location, Age, data source (organic/placed)

Another frustration is that location data is not shared often on profiles
so we are unable to scrape it. Another thing is the upvote feature is not used much
but if it was, it may help indicate a review is of a good quality

multiple picks of the same users will need cleaning up just before analysis
it will be too much work to do here when it is a trivial task in the next stage
"""
from bs4 import BeautifulSoup
import requests
import re #regex
import pandas as pd

extra_reviews_users = []
#range(1,11) = page 1 to 10, 200 reviews in total
for i in range(1,11):
	html = requests.get('https://www.trustpilot.com/review/www.asos.com?page='+str(i)).text
	soup = BeautifulSoup(html,'html5lib')

	user_dicts = []
	review_dicts = []
	supp_dicts = []

	#20 reviews per page
	user_list = []
	user_number_of_reviews = []
	user_profile_links = []

	review_scores = []
	review_datetime = []
	review_titles = []
	review_body = []

	#nested reviews
	supp_scores = []
	supp_datetime = []
	supp_titles = []
	supp_body = []

	#customer names
	for div in soup('div','consumer-information__name'):
		user_list.append(div.text)

	#custmer number of reviews
	for div in soup('div','consumer-information__review-count'):
		for span in div.find_all('span'):
			user_number_of_reviews.append(span.text[0])

	#get links for the user profiles
	for aside in soup('aside','review__consumer-information'):
		for a in aside.find_all('a'):
			user_profile_links.append(a['href'])

	for idx,link in enumerate(user_profile_links):
		user_dicts.append({'userid':link,
			'review_count':user_number_of_reviews[idx],
			'name':user_list[idx]})
		if int(user_number_of_reviews[idx]) > 1:
			extra_reviews_users.append(link)

	df_users = pd.DataFrame(user_dicts)
	df_users.to_csv('user_info.csv',mode='a',header=False)

	#stars given for this individual review
	for entry in soup.find_all(attrs={'class': re.compile('star-rating star-rating-\d')}):
		if len(entry['class']) == 3 and entry['class'][2] == 'star-rating--medium':
			review_scores.append(entry['class'][1][12])

	#title of this individual review
	for h2 in soup('h2','review-content__title'):
		review_titles.append(h2.text)
	
	#body of this review
	for p in soup('p','review-content__text'):
		review_body.append(p.text)

	#get the time and date the review was posted
	for reviewtime in soup('script',attrs={'data-initial-state': re.compile('review-dates')}):
		review_datetime.append(reviewtime.text[19:39])

	for idx,reviewtime in enumerate(review_datetime):
		review_dicts.append({'userid':user_profile_links[idx],
			'rating':review_scores[idx],
		'review title':review_titles[idx],
		'review body':review_body[idx],
		'review datetime':review_datetime[idx]})

	df_reviews = pd.DataFrame(review_dicts)
	df_reviews.to_csv('asos_reviews.csv',mode='a',header=False)

#reviews which have been reported are an issue, since they still show up
#but have no body or title
for link in extra_reviews_users:
	new_html = requests.get('https://www.trustpilot.com'+link).text
	new_soup = BeautifulSoup(new_html,'html5lib')

	#stars given for this individual review
	for entry in new_soup.find_all(attrs={'class': re.compile('star-rating star-rating-\d')}):
		if len(entry['class']) == 3 and entry['class'][2] == 'star-rating--medium':
			supp_scores.append(entry['class'][1][12])

	length_supp_title = len(supp_titles)
	#title of this individual review
	for h2 in new_soup('h2','review-content__title'):
		supp_titles.append(h2.text)

	#if the review was reported we didn't have one and we've got a problem on our hands
	if len(supp_titles) != len(supp_scores):
		supp_titles.append('Reported Review - No title available')

	#body of this review
	length_supp_body = len(supp_body)
	for p in new_soup('p','review-content__text'):
		supp_body.append(p.text)

	if len(supp_body) != len(supp_scores):
		supp_body.append('Reported Review - No Review available')

	#get the time and date the review was posted
	for reviewtime in new_soup('script',attrs={'data-initial-state': re.compile('review-dates')}):
		supp_datetime.append(reviewtime.text[19:39])
	

for idx,score in enumerate(supp_scores):
	supp_dicts.append({'userid':link,
		'rating':score,
		'review title':supp_titles[idx],
		'review body':supp_body[idx],
		'review datetime':supp_datetime[idx]})

df_supp_reviews = pd.DataFrame(supp_dicts)
df_supp_reviews.to_csv('supplementary_reviews.csv',mode='a',header=False)