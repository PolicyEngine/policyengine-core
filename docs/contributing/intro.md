# How to contribute

Any and all contributions are welcome to this project. You can help by:

* Filing issues. Tell us about bugs you've found, or features you'd like to see.
* Fixing issues. File a pull request to fix an issue you or someone else has filed.

If you file an issue or a pull request, one of the maintainers (primarily @nikhilwoodruff) will respond to it within at least a week. If you don't hear back, feel free to ping us on the issue or pull request.

## Changelog Entries

Before you send out a pull request, make sure to add a description of your changes to [changelog_entry.yaml](../../changelog_entry.yaml).
For example,
```yaml
- bump: patch
  changes:
    fixed:
    - Fixed a bug causing Windows tests to fail.
```
You can find more examples in [changelog.yaml](../../changelog.yaml). Note that you do not need to add the date field.
That field is automatically populated by `make changelog`. Also, note that **you should not run `make changelog` 
yourself**, as [our GitHub workflows](../../.github/workflows) will do this for you as part of the build process.

## Pull requests

Each pull request should:
* Close an issue. If there isn't an issue that the pull request completely addresses, please file one.
* Have a description that makes sense to a layperson. If you're fixing a bug, describe what the bug is and how you fixed it. If you're adding a feature, describe what the feature is and why you added it.
* Have tests. If you're fixing a bug, write a test that fails without your fix and passes with it. If you're adding a feature, write tests that cover the feature. Sometimes this isn't necessary (for example, documentation changes), but if in doubt, err on the side of including tests.
* Pass all GitHub actions. If you're not sure why a GitHub action is failing, feel free to ask for help in the issue or pull request.

