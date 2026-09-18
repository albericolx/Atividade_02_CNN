import urllib.request, xml.etree.ElementTree as ET, ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
url = 'https://export.arxiv.org/api/query?search_query=all:CIFAR-10+AND+all:CNN+AND+all:image+AND+all:classification&sortBy=relevance&sortOrder=descending&max_results=5'
response = urllib.request.urlopen(url, context=ctx)
tree = ET.fromstring(response.read())
for entry in tree.findall('{http://www.w3.org/2005/Atom}entry'):
    title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
    print('Title:', title)
    print('Published:', entry.find('{http://www.w3.org/2005/Atom}published').text)
    print('Link:', entry.find('{http://www.w3.org/2005/Atom}id').text)
    print('---')