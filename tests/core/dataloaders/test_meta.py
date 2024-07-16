from giskard_vision.core.dataloaders.meta import MetaData


def test_meta():
    data = {
        "key1": "value1",
        "key2": 123,
        "key3": [1, 2, 3],
        "key4": {"nested_key": "nested_value"},
        "key5": (4, 5),
        "key6": {6, 7},
    }
    categories = ["category1", "category2"]

    # Initialize the MetaData object
    meta = MetaData(data, categories)

    # Test data attribute
    assert meta.data == data, "data attribute test failed"

    # Test categories attribute
    assert meta.categories == categories, "categories attribute test failed"

    # Test get method
    assert meta.get("key1") == "value1", "get method test failed for key1"
    assert meta.get("key2") == 123, "get method test failed for key2"
    try:
        meta.get("nonexistent_key")
    except KeyError as e:
        assert e.args[0] == "Key 'nonexistent_key' not found in the metadata", "get method KeyError test failed"

    # Test get_includes method
    assert meta.get_includes("key1") == "value1", "get_includes method test failed for 'key1'"
    assert meta.get_includes("key2") == 123, "get_includes method test failed for 'key2'"
    try:
        meta.get_includes("nonexistent_substring")
    except KeyError as e:
        assert (
            e.args[0] == "No keys containing 'nonexistent_substring' found in the metadata"
        ), "get_includes method KeyError test failed"
    try:
        meta.get_includes("key")
    except ValueError as e:
        assert (
            e.args[0]
            == "Multiple keys containing 'key' found in the metadata: ['key1', 'key2', 'key3', 'key4', 'key5', 'key6']"
        ), "get_includes method ValueError test failed"

    # Test is_scannable method
    assert MetaData.is_scannable("string"), "is_scannable method test failed for 'string'"
    assert MetaData.is_scannable(123), "is_scannable method test failed for 123"
    assert not MetaData.is_scannable([1, 2, 3]), "is_scannable method test failed for list"
    assert not MetaData.is_scannable({"nested_key": "nested_value"}), "is_scannable method test failed for dict"
    assert not MetaData.is_scannable((4, 5)), "is_scannable method test failed for tuple"
    assert not MetaData.is_scannable({6, 7}), "is_scannable method test failed for set"

    # Test get_scannable method
    expected_scannable = {
        "key1": "value1",
        "key2": 123,
    }
    assert meta.get_scannable() == expected_scannable, "get_scannable method test failed"

    # Test get_categories method
    assert meta.get_categories() == categories, "get_categories method test failed"

    # Test when categories is None
    meta_no_categories = MetaData(data)
    assert meta_no_categories.get_categories() is None, "get_categories method test failed for None categories"
