"""
Milvus API 测试脚本

测试两个新增的 API 接口：
1. POST /api/v1/milvus/collections/delete - 删除集合
2. POST /api/v1/milvus/ingest/file - 文件入库
"""
import requests
import json

BASE_URL = "http://localhost:8000"
API_KEY = "test"

def test_delete_collection():
    """测试删除 Milvus 集合接口"""
    print("=" * 60)
    print("测试 1: 删除 Milvus 集合")
    print("=" * 60)
    
    url = f"{BASE_URL}/api/v1/milvus/collections/delete"
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "collection_name": "test_collection_delete"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"状态码：{response.status_code}")
        print(f"响应内容：{json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            print("✅ 删除集合成功")
        else:
            print(f"❌ 删除集合失败：{response.text}")
    except Exception as e:
        print(f"❌ 请求失败：{e}")
    
    print()


def test_ingest_file():
    """测试文件入库接口"""
    print("=" * 60)
    print("测试 2: 文件入库")
    print("=" * 60)
    
    url = f"{BASE_URL}/api/v1/milvus/ingest/file"
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "file_path": "", # 知识库路径
        "collection_name": "test_collection_ingest"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"状态码：{response.status_code}")
        print(f"响应内容：{json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            print("✅ 文件入库成功")
        else:
            print(f"❌ 文件入库失败：{response.text}")
    except Exception as e:
        print(f"❌ 请求失败：{e}")
    
    print()


def test_ingest_nonexistent_file():
    """测试文件不存在的情况"""
    print("=" * 60)
    print("测试 3: 文件不存在（预期失败）")
    print("=" * 60)
    
    url = f"{BASE_URL}/api/v1/milvus/ingest/file"
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "file_path": "/non/existent/file.txt",
        "collection_name": "test_collection"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"状态码：{response.status_code}")
        print(f"响应内容：{json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        if response.status_code == 400:
            print("✅ 正确返回 400 错误")
        else:
            print(f"❌ 预期返回 400，实际：{response.status_code}")
    except Exception as e:
        print(f"❌ 请求失败：{e}")
    
    print()


def test_unauthorized():
    """测试未授权访问"""
    print("=" * 60)
    print("测试 4: 未授权访问（预期失败）")
    print("=" * 60)
    
    url = f"{BASE_URL}/api/v1/milvus/collections/delete"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "collection_name": "test_collection"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"状态码：{response.status_code}")
        
        if response.status_code == 401:
            print("✅ 正确返回 401 未授权错误")
        else:
            print(f"❌ 预期返回 401，实际：{response.status_code}")
    except Exception as e:
        print(f"❌ 请求失败：{e}")
    
    print()


if __name__ == "__main__":
    print("\n开始测试 Milvus API 接口...\n")
    
    # 测试各个接口
    test_unauthorized()  # 测试鉴权
    test_delete_collection()  # 测试删除集合
    test_ingest_file()  # 测试文件入库
    test_ingest_nonexistent_file()  # 测试文件不存在
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)
