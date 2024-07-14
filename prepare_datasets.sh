gdown 1CV42PDFY_l5hvk492llmMej-5cG_D7TA
gdown 1ySR2e7r2BrYLjJgmqwwAWewLYq_dYvGv
unzip drive-download-20221130T095944Z-001.zip -d datasets
unzip drive-download-20221130T095944Z-002.zip -d datasets

cd datasets

tar -xvf amazon_review_full_csv.tar.gz -C amazon_review_full
tar -xvf amazon_review_polarity_csv.tar.gz -C amazon_review_polarity
tar -xvf ag_news_csv.tar.gz -C ag_news
tar -xvf dbpedia_csv.tar.gz -C dbpedia
tar -xvf sogou_news_csv.tar.gz -C sogou_news
tar -xvf yahoo_answers_csv.tar.gz -C yahoo_answers
tar -xvf yelp_review_full_csv.tar.gz -C yelp_review_full
tar -xvf yelp_review_polarity_csv.tar.gz -C yelp_review_polarity
mv amazon_review_full/amazon_review_full_csv/* amazon_review_full
mv amazon_review_polarity/amazon_review_polarity_csv/* amazon_review_polarity
mv ag_news/ag_news_csv/* ag_news
mv dbpedia/dbpedia_csv/* dbpedia
mv sogou_news/sogou_news_csv/* sogou_news
mv yahoo_answers/yahoo_answers_csv/* yahoo_answers
mv yelp_review_full/yelp_review_full_csv/* yelp_review_full
mv yelp_review_polarity/yelp_review_polarity_csv/* yelp_review_polarity
rm -r amazon_review_full/amazon_review_full_csv
rm -r amazon_review_polarity/amazon_review_polarity_csv
rm -r ag_news/ag_news_csv
rm -r dbpedia/dbpedia_csv
rm -r sogou_news/sogou_news_csv
rm -r yahoo_answers/yahoo_answers_csv
rm -r yelp_review_full/yelp_review_full_csv
rm -r yelp_review_polarity/yelp_review_polarity_csv