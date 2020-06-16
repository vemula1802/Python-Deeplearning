import requests
from bs4 import BeautifulSoup

class WebScrapper():
   wiki = requests.get("https://en.wikipedia.org/wiki/Deep_learning") #Storing all the data into wiki
   soup = BeautifulSoup(wiki.content, "html.parser") # Getting the html content
   print(soup)

   def getTitle(self):
       title = self.soup.title.string  #Retreiving the title
       return title

   def getWikiLinks(self):
       list = [] #Creating a list to store all the href links
       for link in self.soup.find_all('a'):
           list.append(link.get('href'))
       return list

obj = WebScrapper() #Object for the created class to call its functions
print(obj.getTitle())
for x in obj.getWikiLinks():
    print ( x, file=open("output.txt", "a"))
