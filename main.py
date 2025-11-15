from dotenv import load_dotenv

from rag import create_rag_service


# ---------------------------------------------------------------------
# RAG Chatbot Setup
# ---------------------------------------------------------------------
def main() -> None:
    load_dotenv()
    print("-" * 60)
    print("RAG Chatbot - Ask questions about your documents!")
    print("-" * 60)
    print()

    try:
        rag_service = create_rag_service()
        print("Chatbot Ready!")
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    while True:
        user_input = input("\nYou: ").strip()

        # Exit condition
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting the chatbot. Goodbye!")
            break

        try:
            result = rag_service.ask(user_input)
            print(f"\nAI: {result.answer}")

        except Exception as e:
            print(f"An error occurred while processing your request: {e}")

if __name__ == "__main__":
    main()

