import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Téléchargement du modèle GPT-2
model_name = "gpt2"  # Vous pouvez utiliser d'autres modèles GPT-2 en spécifiant leur nom
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Interface Streamlit
st.title("Déploiement d'un modèle GPT-2 avec Streamlit")

# Saisie de l'utilisateur
user_input = st.text_input("Entrez du texte :")

# Génération de texte avec GPT-2
if user_input:
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write("Texte généré par GPT-2 :")
    st.write(generated_text)

