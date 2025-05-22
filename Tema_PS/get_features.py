from openai import OpenAI
import pyperclip
import time

openai_api_key = "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"  # Replace this with your actual API key

client = OpenAI(api_key=openai_api_key)
pyperclip.ENCODING = "utf-8"

def complete(prompt):
    database_description = (
        "Aceasta este o bază de date relațională pentru gestionarea comenzilor unei companii. "
        "Conține următoarele tabele principale:\n"
        "- Region: stochează informații despre regiuni, cu coloane RegionID (cheie primară) și RegionDescription.\n"
        "- Territories: descrie teritoriile, cu TerritoryID (cheie primară), TerritoryDescription și RegionID (cheie externă).\n"
        "- EmployeeTerritories: relaționează angajații cu teritoriile lor, cu EmployeeID și TerritoryID (chei externe).\n"
        "- Employees: informații despre angajați (e.g., EmployeeID, LastName, FirstName, etc.).\n"
        "- Orders: conține informații despre comenzi (e.g., OrderID, CustomerID, EmployeeID, etc.).\n"
        "- Order Details: detalii despre produse din comenzi (OrderID, ProductID, UnitPrice, etc.).\n"
        "- Products: detalii despre produse (ProductID, ProductName, CategoryID, etc.).\n"
        "- Suppliers: informații despre furnizori (SupplierID, CompanyName, etc.).\n"
        "- Categories: categorii de produse (CategoryID, CategoryName, etc.).\n"
        "- Customers: informații despre clienți (CustomerID, CompanyName, etc.).\n"
        "- Shippers: detalii despre companiile de transport (ShipperID, CompanyName, etc.).\n"
        "Toate relațiile între tabele sunt configurate cu chei externe corespunzătoare. "
        "Scrie un cod PL/SQL care să răspundă cerințelor specificate mai jos. "
        "Raspunde doar cu cod, pune comentarii in el, dar doar atat, fara alte explicatii"
    )

    result = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant skilled in PL/SQL and databases."},
            {
                "role": "user",
                "content": database_description + "\n" + prompt
            }
        ]
    )
    return result.choices[0].message.content


def check_exit_condition(content):
    if content == "x":
        pyperclip.copy("x")
        exit()

def should_process_clipboard(current_clipboard_content, previous_clipboard_content, previous_response):
    return (
        current_clipboard_content != previous_clipboard_content and
        current_clipboard_content != previous_response and
        current_clipboard_content.strip()
    )

def handle_clipboard_content(content):
    response = complete(content)
    pyperclip.copy(response)
    return response


def main():
    previous_clipboard_content = ""
    previous_response = ""
    while True:
        current_clipboard_content = pyperclip.paste()

        if should_process_clipboard(current_clipboard_content, previous_clipboard_content, previous_response):
            previous_clipboard_content = current_clipboard_content
            previous_response = handle_clipboard_content(current_clipboard_content)

        check_exit_condition(current_clipboard_content)

        time.sleep(1)

if __name__ == "_main_":
    main()