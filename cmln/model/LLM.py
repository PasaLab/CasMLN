import torch

import os
import openai

openai.api_key = "xxxx"
openai.api_base="xxxx"
# os.environ["OPENAI_API_BASE"] = "xxxx"
# os.environ["OPENAI_API_KEY"] = "xxxx"

class LLM_model(torch.nn.Module):
    def __init__(
        self,
        hid_dim,
        dataset,
        device
    ):
        super().__init__()
        self.dataset = dataset
        self.device = device

    def graph_encode(self):
        if self.dataset == "Aminer":
            text = "Aminer is a temporal heterogeneous graph dataset about academic citations. Its time slices are separated using the publication year (during 1990-2006) of papers. The graph consists of three types of nodes (paper, author and venue), and two types of relations (paper-publish-venue and author-writer-paper). "
        elif self.dataset == "Ecomm":
            text = "Ecomm is a real-world temporal heterogeneous bipartite graph of the ecommerce, which mainly records shopping behaviors of users within 11 daily snapshots from 10th June 2019 to 20th June 2019. It consists of two types of nodes (user and item) and four types of relations (user-click-item, user-buy-item, user-(add-to-cart)-item and user-(add-to-favorite)-item)."
        elif self.dataset == "Yelp-nc":
            text = "Yelp is a business review net containing timestamped user reviews and tips on businesses. There are two types of nodes (users and business) and two types of edges (user-tip-business and userreview-business) in the temporal heterogeneous graph constructed based on it."
        elif self.dataset == "covid":
            text = "COVID-19 is an epidemic disease dataset, which contains both state and county level daily COVID-19 case reports in the US. It includes 304 graph slices spanning from 05/01/2020 to 02/28/2021. Each graph slice is also a heterogeneous graph consisting of two types of nodes (state and county) and three types of relations between them (state-includes-county, state-near-state, and county-near-county)."
        request = "Please output a summary of the information about this temporal heterogeneous graph in the following format: {Domain:,Introduction:,Node types:,Relations:}."
        summary = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text+request}
            ]
        ).choices[0].message.content
        # print(summary)
        emb = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=summary,
                )["data"][0]["embedding"]
        emb = torch.tensor(emb)
        emb = emb.to(self.device)
        return emb
    
    def cate_encode(self):
        if self.dataset == "Aminer":
            text = ["Academic papers that may be published in venues.","Authors who may write papers.","Venues for the publication of papers."]
        elif self.dataset == "Ecomm":
            text = ["Ecommerce users who may interact (click, buy, cart, favorite) with items.", "Ecommerce items that may be clicked, bought, added to cart, or favorited by users."]
        elif self.dataset == "Yelp-nc":
            text = ["Users of the business review net Yelp. They may review or tip businesses.","Businesses of the business review net Yelp. They may be reviewed or tipped by users."]
        elif self.dataset == "covid":
            text = ["States of the US that may be near other states or include counties.","Counties of the US that may be near other counties or be included by states."]
        request = "Please output a summary of the information about this node type in the following format: {Introduction:,Relevant relations:}."
        embeddings = []

        for t in text:
            summary = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": t+request}
                ]
            ).choices[0].message.content
            # print(summary)

            emb = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=summary,
                )["data"][0]["embedding"]
            emb = torch.tensor(emb)
            emb = emb.to(self.device)
            embeddings.append(emb)

        return embeddings