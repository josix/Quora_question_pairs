from Parser import Parser

def Preprocessor(s):
	
	parser = Parser()
	tokens = parser.tokenise(s)
	#result = parser.removeStopWords(tokens)
	return " ".join(tokens)
	
if __name__ == '__main__':
	print (Preprocessor("dogs calls the cats"))


