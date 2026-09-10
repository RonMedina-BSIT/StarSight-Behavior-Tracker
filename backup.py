import firebase_admin
from firebase_admin import credentials, firestore
import json
from datetime import datetime

# 1. Authenticate using your existing credentials
cred = credentials.Certificate('firebase-admin.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

def backup_firestore():
    print("Starting Firestore backup...")
    backup_data = {}
    
    # 2. Fetch all users
    users_ref = db.collection('users')
    users = users_ref.stream()

    for user in users:
        user_data = user.to_dict()
        
        # 3. Fetch the nested 'children' subcollection for this user
        children_ref = users_ref.document(user.id).collection('children')
        children = children_ref.stream()

        children_data = {}
        for child in children:
            children_data[child.id] = child.to_dict()

        # Append children to the user's data profile
        user_data['children'] = children_data
        backup_data[user.id] = user_data

    # 4. Save to a timestamped JSON file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"firestore_backup_{timestamp}.json"
    
    with open(filename, 'w') as f:
        # default=str handles Firestore Timestamp objects safely
        json.dump(backup_data, f, default=str, indent=4) 
        
    print(f"Success! Backup saved locally as: {filename}")

if __name__ == '__main__':
    backup_firestore()