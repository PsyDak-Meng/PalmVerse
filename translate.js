async function translateText(text) {
    try {
      const response = await axios.post(
        `https://translation.googleapis.com/language/translate/v2?key=AIzaSyC449Bf8vA6Em3vV62cdMRGaKO6gk-8hiU`,
        { q: text, target: "tr" }
      );
  
      const translation = response.data.data.translations[0].translatedText;
      return translation;
    } catch (error) {
      console.error('Error translating text:', error);
      return null; // or handle the error in a different way
    }
  }
  
  // Example usage
  const translatedText = await translateText("Hello, World!");
  console.log(translatedText);
  